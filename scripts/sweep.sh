#!/usr/bin/env bash
#
# Watcher: every pass, hand every run that has no result file yet to a slurm array task, until the
# deadline. One array task per run (combo x seed) on one cpu, so a short wall clock fits every task
# and preempted work costs only that seed; the persistent jax cache carries compilation across
# tasks. The queue caps how many tasks may sit submitted, so each pass tops the arrays back up.
# Watchers started under different names keep their own log and manifests, so two of them never
# hand out the same run.
#
# usage: sweep.sh [name] [deadline_hours]

sweep=${1:-main}
deadline_hours=${2:-72}

# archived campaigns each keep their own tree under results/
RESULTS_DIR=${RESULTS_DIR:-results/neurips}

################################################################################
# WHAT TO RUN

# memory and wall clock clear the slowest per-seed run on record with room for a cold compile, and
# no more: a shorter wall backfills sooner, but one that a run overruns loses the seed and earns it
# back next pass
declare -A TARGET_RESOURCES=(
    [gramacylee]="8G 0:30:00"
    [ackley]="8G 0:30:00"
    [hartmann]="8G 0:30:00"
    [rosenbrock]="8G 0:30:00"
    [michalewicz]="8G 1:00:00"
    [pendulum]="8G 1:30:00"
    [pinwheel]="8G 1:30:00"
    [brachistochrone]="8G 1:30:00"
    [hopper]="8G 1:30:00"
    [mnist]="24G 06:00:00"
)

# the budget goes in target order, so the short seeds are scheduled first
targets=(
    brachistochrone
    pinwheel
    gramacylee
    ackley
    hartmann
    rosenbrock
    michalewicz
    pendulum
    hopper
    mnist
)

variants=(
    "random"
    "ours"
    "ours_adaptive"
    "vien"
    "shilton"
    "vellanky"
    "kundu"
    "vien --disable_natural_gradient"
    "ours --disable_natural_gradient"
    "ours_adaptive --disable_natural_gradient"
    "ours --constrain_order"
    "ours --disable_natural_gradient --constrain_order"
    "ours --lstsq_rcond=1e-10"
    "ours --lstsq_rcond=1e-08"
    "ours --lstsq_rcond=1e-06"
    "ours --sample_candidates_from_gp"
    "shilton --reduced_grid"
    "ours --fixed_k=1"
    "ours --fixed_k=5"
    "ours --fixed_k=10"
)

lengthscales=(0.4 0.2 0.1 0.05)
profiles=(rbf matern52 matern32 matern12)
seeds=($(seq 0 31))

# qos normal caps a user at 2000 submitted and 1000 running (sacctmgr show qos), and the tally in
# the main loop counts every job of ours, so filling the cap exactly is safe: a refused sbatch is
# retried next pass
MAX_SUBMITTED=2000
MAX_ARRAY_SIZE=1000

LOG="sweep_${sweep}.log"
MANIFESTS=".sweep_manifests_${sweep}"
DEADLINE=$(( $(date +%s) + deadline_hours * 3600 ))

mkdir -p "$MANIFESTS" logs/

################################################################################
# RUNS

# vellanky's parametrization is 1d only, and the permutation-simplex ablation is defined for the
# ridges and the physical targets, so neither variant runs on every target
MULTIDIM_FNS="ackley|hartmann|rosenbrock|michalewicz|pendulum"
CONSTRAIN_ORDER_FNS="gramacylee|ackley|hartmann|rosenbrock|michalewicz|brachistochrone|hopper|pendulum"

variant_runs_on() {
    local variant=$1 target_fn=$2 lengthscale=$3
    [[ "${variant%% *}" == "vellanky" && "$target_fn" =~ ^($MULTIDIM_FNS)$ ]] && return 1
    [[ "$variant" == *"--constrain_order"* && ! "$target_fn" =~ ^($CONSTRAIN_ORDER_FNS)$ ]] && return 1
    # ours_adaptive searches the whole lengthscale range itself, one run covers the sweep
    [[ "${variant%% *}" == "ours_adaptive" && "$lengthscale" != "0.2" ]] && return 1
    return 0
}

# suffixes accumulate in the same order bofus-run builds save_dir, so combined flags agree
variant_to_dir() {
    local variant=$1 dir=${1%% *}
    [[ "$variant" == *"--disable_natural_gradient"* ]] && dir+="_no_natural_grad"
    [[ "$variant" == *"--sample_candidates_from_gp"* ]] && dir+="_sample_from_gp"
    [[ "$variant" == *"--reduced_grid"* ]] && dir+="_reduced_grid"
    [[ "$variant" == *"--constrain_order"* ]] && dir+="_constrain_order"
    # bofus-run formats these suffixes with %.0e, so the flags are written in that form already,
    # zero-padded exponent included
    [[ "$variant" =~ --lstsq_rcond=([0-9.e+-]+) ]] && dir+="_rcond_${BASH_REMATCH[1]}"
    [[ "$variant" =~ --fixed_k=([0-9]+) ]] && dir+="_fixed_k_${BASH_REMATCH[1]}"
    echo "$dir"
}

# a run is one task's whole workload, so it carries the variant directory the task needs to name
# its own result file rather than re-deriving the mapping inside the job; the seed varies slowest,
# so a capped submission covers the whole grid at low seed counts rather than finishing every seed
# of a few configs and none of the rest
declare -A TARGET_RUNS=() VARIANT_DIR=()
for target_fn in "${targets[@]}"; do
    runs=()
    for seed in "${seeds[@]}"; do
    for profile in "${profiles[@]}"; do
    for lengthscale in "${lengthscales[@]}"; do
    for variant in "${variants[@]}"; do
        variant_runs_on "$variant" "$target_fn" "$lengthscale" || continue
        variant_dir=${VARIANT_DIR[$variant]:=$(variant_to_dir "$variant")}
        runs+=("$profile $lengthscale $variant_dir $seed $variant")
    done
    done
    done
    done
    TARGET_RUNS[$target_fn]=$(printf '%s\n' "${runs[@]}")
done

################################################################################
# QUEUE

# a queued task has no result yet but must not be handed out twice, so a manifest line stands in
# for work in flight only while its own array task lives; manifests of finished arrays are dropped
declare -A INFLIGHT=()
refresh_inflight() {
    INFLIGHT=()
    local live task_state
    live=$(squeue --me -h -o "%F" --states=RUNNING,PENDING,COMPLETING 2>/dev/null | sort -u)
    task_state=$(squeue --me -h -r --states=RUNNING,PENDING,COMPLETING -o "%i" 2>/dev/null | sort -u)

    for jobid_file in "$MANIFESTS"/*.jobid; do
        [[ -e "$jobid_file" ]] || continue
        local manifest="${jobid_file%.jobid}"
        local jobid
        jobid=$(<"$jobid_file")
        if ! grep -qx "$jobid" <<< "$live"; then
            rm -f "$jobid_file" "$manifest"
            continue
        fi
        # a manifest holds one target's runs with the target itself stripped off, so put it back
        # to match the keys the submit scan looks up
        local manifest_target
        manifest_target=$(basename "$manifest")
        manifest_target="${manifest_target%_*}"
        # a plain job claims its whole manifest; an array claims a line only while task N-1 lives,
        # so a cancelled task hands its run straight back instead of waiting out its siblings
        local plain_job=0
        grep -qx "$jobid" <<< "$task_state" && plain_job=1
        local line=0
        while read -r run; do
            if (( plain_job )) || grep -qx "${jobid}_$((line))" <<< "$task_state"; then
                INFLIGHT["$manifest_target $run"]=1
            fi
            (( line++ ))
        done < "$manifest"
    done
}

# how many tasks the last submit_sweep put in the queue; an exit status would wrap at 255
SUBMITTED_N=0

# a run earns a task while its result file is missing, so a job cut short by the wall or preempted
# is picked up again on a later pass
submit_sweep() {
    local target_fn=$1 budget=$2
    SUBMITTED_N=0

    local memory time
    read -r memory time <<< "${TARGET_RESOURCES[$target_fn]}"

    # the scan walks every run of the target on every pass, so the result path is built inline:
    # a command substitution here is a fork per run and costs minutes over the whole sweep
    local runs=() profile lengthscale variant_dir seed
    while read -r run; do
        read -r profile lengthscale variant_dir seed _ <<< "$run"
        [[ -e "$RESULTS_DIR/$target_fn/$variant_dir/${profile}_lengthscale_${lengthscale}/seed_${seed}.pkl" ]] && continue
        [[ -n "${INFLIGHT["$target_fn $run"]:-}" ]] && continue
        runs+=("$run")
        (( ${#runs[@]} >= budget )) && break
    done <<< "${TARGET_RUNS[$target_fn]}"

    local n=${#runs[@]}
    (( n == 0 )) && return 0

    local manifest="$MANIFESTS/${target_fn}_$(date +%s%N)"
    printf '%s\n' "${runs[@]}" > "$manifest"

    local jobid
    jobid=$(sbatch --parsable --job-name="sweep_${target_fn}" \
                   --array=0-$((n - 1)) <<EOF
#!/usr/bin/env bash
#SBATCH --output=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${memory}
#SBATCH --time=${time}
# preempt holds ten times the free cores of cpu, and a task killed there loses at most the seed it
# was on, so let slurm start the array wherever it can and requeue whatever the node owner reclaims
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --requeue

read -r profile lengthscale variant_dir seed method extra_flags \
    <<< "\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")"

# compiled executables bake in host cpu features, so cap the isa and key the cache on it
ISA=\$(grep -qm1 avx512f /proc/cpuinfo && echo AVX512 || echo AVX2)
export XLA_FLAGS="\${XLA_FLAGS:-} --xla_cpu_max_isa=\$ISA --xla_cpu_multi_thread_eigen=false"
export JAX_COMPILATION_CACHE_DIR="\${SLURM_SUBMIT_DIR:-\$PWD}/.jaxcache/\$(uname -m)_\$ISA"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.2
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "\$JAX_COMPILATION_CACHE_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NPY_NUM_THREADS=1

# a requeued task may find its seed already done
export RESULTS_DIR=${RESULTS_DIR}
result=${RESULTS_DIR}/${target_fn}/\${variant_dir}/\${profile}_lengthscale_\${lengthscale}/seed_\${seed}.pkl
[[ -e "\$result" ]] && exit 0

log_dir=logs/${target_fn}/\${variant_dir}_\${profile}_\${lengthscale}
mkdir -p "\$log_dir"

uv run bofus-run \
    --method=\$method \
    --profile=\$profile \
    --target_fn=${target_fn} \
    --lengthscale=\$lengthscale \
    --seed=\$seed \
    --initial_acquisitions=10 \
    --minimum_k=1 \
    --maximum_k=10 \
    --acquisitions_each_k=10 \
    --acquisition_raw_samples=1024 \
    --acquisition_max_restarts=16 \
    \$extra_flags > "\$log_dir/seed_\${seed}.out" 2>&1
EOF
    )

    if [[ -z "$jobid" ]]; then
        echo "[$(date)] $target_fn: sbatch failed for $n runs" | tee -a "$LOG"
        rm -f "$manifest"
        return 0
    fi

    echo "$jobid" > "$manifest.jobid"
    echo "[$(date)] $target_fn: submitted array $jobid with $n runs" | tee -a "$LOG"
    SUBMITTED_N=$n
}

################################################################################
# MAIN LOOP

echo "[$(date)] Watcher started (PID $$), deadline in ${deadline_hours}h at $(date -d @$DEADLINE)" | tee -a "$LOG"

while true; do
    if (( $(date +%s) >= DEADLINE )); then
        echo "[$(date)] ${deadline_hours}h elapsed, exiting." | tee -a "$LOG"
        exit 0
    fi

    refresh_inflight

    # array tasks each count against the submit cap, so -r expands them into the tally
    submitted=$(squeue --me -h -r --states=RUNNING,PENDING,COMPLETING 2>/dev/null | wc -l)
    budget=$(( MAX_SUBMITTED - submitted ))

    # each target takes all the budget it can use before the next one is offered any, so the short
    # targets are fully scheduled ahead of the rest and the last one gets what nothing else claimed
    for target_fn in "${targets[@]}"; do
        (( budget <= 0 )) && break
        share=$budget
        (( share > MAX_ARRAY_SIZE )) && share=$MAX_ARRAY_SIZE
        submit_sweep "$target_fn" "$share"
        (( budget -= SUBMITTED_N ))
    done

    sleep 600
done

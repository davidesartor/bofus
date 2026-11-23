#!/usr/bin/env bash
# Usage: [SLURM=1] [TARGETS="a b" TIER=name] [PACK=n] [DISTANCES="rkhs l2"] [PARTITION=gpu GRES=gpu:1080_ti:1 RUNNER=<cuda python>] sweep.sh [closed] [open] [synthetic]  (default: all three, in that order)
# One job per (target, distance, seed, m); seeds are innermost, so PACK=SEEDS puts
# every seed of a config on one task. Complete runs are skipped, interrupted ones resume.
set -euo pipefail

SEEDS=${SEEDS:-8}
PROFILE=${PROFILE:-matern52}
M=${M:-8}  # space separated basis sizes, e.g. M="1 2 4 8"
INITIAL=${INITIAL:-4}
BUDGET=${BUDGET:-256}
JOBS=${JOBS:-10}
DISTANCES=${DISTANCES:-rkhs l2}
PACK=${PACK:-1}  # runs sharing one slurm task (and gpu)
RESULTS_DIR=${RESULTS_DIR:-results/tbd/sweep}
RUN_ARGS=${RUN_ARGS:-}  # extra run.py flags, e.g. RUN_ARGS=--atoms_per_dim=1
TAG=${RUN_ARGS//[ -]/}

SYNTHETIC=(
    gramacylee_ridge ackley_ridge hartmann_ridge rosenbrock_ridge michalewicz_ridge
    gramacylee_projection ackley_projection hartmann_projection
    rosenbrock_projection michalewicz_projection
)
# sorted by action dimension, then simulation cost
ENVS=(
    Pendulum-v1 MountainCarContinuous-v0 InvertedPendulum-v5 InvertedDoublePendulum-v5
    Reacher-v5 LunarLanderContinuous-v3 Swimmer-v5 Hopper-v5
    BipedalWalker-v3 BipedalWalkerHardcore-v3 Walker2d-v5 HalfCheetah-v5 Pusher-v5
    Ant-v5 Humanoid-v5 HumanoidStandup-v5
)

suites=("$@")
((${#suites[@]})) || suites=(closed open synthetic)
targets=()
# TARGETS picks targets by name across the suites, TIER names that subset's list and jobs
TIER=${TIER:-}
for suite in "${suites[@]}"; do
    case "$suite" in
        synthetic) targets+=("${SYNTHETIC[@]}") ;;
        open) targets+=("${ENVS[@]/%/-open-loop}") ;;
        closed) targets+=("${ENVS[@]}") ;;
        *) echo "unknown suite: $suite" >&2 && exit 1 ;;
    esac
done
if [[ -n "${TARGETS:-}" ]]; then
    read -ra targets <<< "$TARGETS"
fi

# one process per core, each pinned to a single thread
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# cuda graphs need driver memory outside the pools that packed runs keep at their peak;
# unpacked runs capture whole while loops in graphs, so L-BFGS-B steps skip host round trips
if ((PACK > 1)); then
    XLA_FLAGS+=" --xla_gpu_enable_command_buffer="
else
    XLA_FLAGS+=" --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PROFILE M INITIAL BUDGET RESULTS_DIR PACK RUN_ARGS TAG

job() {
    local target=$1 distance=$2 seed=$3 m=$4
    local log="$RESULTS_DIR/logs/${target}_${distance}${TAG:+_$TAG}_${PROFILE}_${seed}_m${m}.log"
    ${RUNNER:-uv run} scripts/run.py --target_fn "$target" --profile "$PROFILE" \
        --distance "$distance" --seed "$seed" --m "$m" \
        --initial_acquisitions "$INITIAL" --budget "$BUDGET" $RUN_ARGS \
        --results_dir "$RESULTS_DIR" --verbose > "$log" 2>&1 \
        && echo "done: $target $distance $seed $m" \
        || echo "FAILED: $target $distance $seed $m"
}
export -f job

mkdir -p "$RESULTS_DIR/logs"
# one list per (suites, sizes), so a new submission never rewrites a queued array's list
jobs_file="$RESULTS_DIR/jobs_${TIER:-$(IFS=-; echo "${suites[*]}")}_m${M// /-}_${DISTANCES// /-}${TAG:+_$TAG}.txt"

# inside a slurm array task: run its lines of the job list, on the synced venv
# directly since hundreds of concurrent uv invocations race on its cache;
# requeued 10 min before the time limit, complete runs are skipped and interrupted ones resume
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export RUNNER=${RUNNER:-.venv/bin/python}
    first=$(((SLURM_ARRAY_TASK_ID - 1) * PACK + 1))
    set -m  # the pool gets its own process group, killed whole on requeue
    trap 'kill -- -"$pool" 2>/dev/null; scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; exit 0' USR1
    xargs -d '\n' -n 1 -P "$PACK" -a <(sed -n "${first},$((first + PACK - 1))p" "$jobs_file") bash -c 'job $1' _ &
    pool=$!
    wait "$pool" || true
    exit
fi

for m in $M; do
    for target in "${targets[@]}"; do
        for distance in $DISTANCES; do
            for seed in $(seq 0 $((SEEDS - 1))); do
                echo "$target $distance $seed $m"
            done
        done
    done
done > "$jobs_file.tmp"
mv "$jobs_file.tmp" "$jobs_file"  # atomic, running tasks may be reading it

# with SLURM=1 submit the list as a single-cpu array (ARRAY=a-b submits a chunk), else run a local pool
if [[ "${SLURM:-0}" == 1 ]]; then
    uv sync --inexact  # keeps a separately installed CUDA jax plugin
    sbatch --array="${ARRAY:-1-$(( ($(wc -l < "$jobs_file") + PACK - 1) / PACK ))}" --partition="${PARTITION:-cpu}" ${GRES:+--gres="$GRES"} ${CONSTRAINT:+--constraint="$CONSTRAINT"} \
        --cpus-per-task="$PACK" --mem="$((PACK > 8 ? 2 * PACK : 16))G" --time="${TIME:-2-00:00:00}" --signal=B:USR1@600 --requeue \
        --job-name="bofus-sweep${TIER:+-$TIER}" \
        --output="$RESULTS_DIR/logs/slurm_%A_%a.out" scripts/sweep.sh "${suites[@]}"
else
    tr '\n' '\0' < "$jobs_file" | xargs -0 -n 1 -P "$JOBS" bash -c 'job $1' _
fi

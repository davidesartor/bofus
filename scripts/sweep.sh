#!/usr/bin/env bash
# Usage: [SLURM=1] sweep.sh [synthetic] [open] [closed]  (default: all three)
# One pool of JOBS single-threaded workers; jobs go seed-major so every target gets
# seed 0 first, gym envs in order of complexity. Complete runs are skipped by run.py.
set -euo pipefail

SEEDS=${SEEDS:-4}
PROFILE=${PROFILE:-matern52}
M=${M:-8}
INITIAL=${INITIAL:-4}
BUDGET=${BUDGET:-128}
JOBS=${JOBS:-10}
RESULTS_DIR=${RESULTS_DIR:-results/sweep}

SYNTHETIC=(
    gramacylee_ridge ackley_ridge hartmann_ridge rosenbrock_ridge michalewicz_ridge
    gramacylee_projection ackley_projection hartmann_projection
    rosenbrock_projection michalewicz_projection
)
# sorted by action dimension, then simulation cost; large = closed-loop d >= 105
SMALL_ENVS=(
    Pendulum-v1 MountainCarContinuous-v0 InvertedPendulum-v5 InvertedDoublePendulum-v5
    Reacher-v5 LunarLanderContinuous-v3 Swimmer-v5 Hopper-v5
    BipedalWalker-v3 BipedalWalkerHardcore-v3 Walker2d-v5 HalfCheetah-v5 Pusher-v5
)
LARGE_ENVS=(Ant-v5 Humanoid-v5 HumanoidStandup-v5)
ENVS=("${SMALL_ENVS[@]}" "${LARGE_ENVS[@]}")

suites=("$@")
((${#suites[@]})) || suites=(synthetic open closed)
targets=()
for suite in "${suites[@]}"; do
    case "$suite" in
        synthetic) targets+=("${SYNTHETIC[@]}") ;;
        open) targets+=("${ENVS[@]/%/-open-loop}") ;;
        closed) targets+=("${SMALL_ENVS[@]}") ;;
        *) echo "unknown suite: $suite" >&2 && exit 1 ;;
    esac
done

# one process per core, each pinned to a single thread
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PROFILE M INITIAL BUDGET RESULTS_DIR

job() {
    local target=$1 distance=$2 seed=$3
    local log="$RESULTS_DIR/logs/${target}_${distance}_${PROFILE}_${seed}.log"
    ${RUNNER:-uv run} scripts/run.py --target_fn "$target" --profile "$PROFILE" \
        --distance "$distance" --seed "$seed" --m "$M" \
        --initial_acquisitions "$INITIAL" --budget "$BUDGET" \
        --results_dir "$RESULTS_DIR" --verbose > "$log" 2>&1 \
        && echo "done: $target $distance $seed" \
        || echo "FAILED: $target $distance $seed"
}
export -f job

mkdir -p "$RESULTS_DIR/logs"
jobs_file="$RESULTS_DIR/jobs.txt"

# inside a slurm array task: run its line of the job list, on the synced venv
# directly since hundreds of concurrent uv invocations race on its cache
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUNNER=.venv/bin/python job $(sed -n "${SLURM_ARRAY_TASK_ID}p" "$jobs_file")
    exit
fi

for seed in $(seq 0 $((SEEDS - 1))); do
    for target in "${targets[@]}"; do
        for distance in rkhs l2; do
            echo "$target $distance $seed"
        done
    done
done > "$jobs_file"

# with SLURM=1 submit the list as a single-cpu array, else run a local pool
if [[ "${SLURM:-0}" == 1 ]]; then
    uv sync
    sbatch --array="1-$(wc -l < "$jobs_file")" --partition="${PARTITION:-cpu}" \
        --cpus-per-task=1 --mem=16G --time="${TIME:-2-00:00:00}" --job-name=bofus-sweep \
        --output="$RESULTS_DIR/logs/slurm_%A_%a.out" scripts/sweep.sh "${suites[@]}"
else
    tr '\n' '\0' < "$jobs_file" | xargs -0 -n 1 -P "$JOBS" bash -c 'job $1' _
fi

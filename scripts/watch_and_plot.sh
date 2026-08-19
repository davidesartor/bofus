#!/usr/bin/env bash
# Wait for the sweep to drain, then rebuild the CSVs and the figures.

cd "$(dirname "$0")/.." || exit 1

LOG="watch_and_plot.log"
CONVERTER="/home/dsartor_umass_edu/BOFUS-pickle/scripts/convert_pickles.py"
QUIET_CHECKS_NEEDED=3  # consecutive empty polls, so a gap between arrays is not the end
POLL_SECONDS=600
DEADLINE=$(( $(date +%s) + ${1:-80} * 3600 ))

log() { echo "[$(date)] $*" >> "$LOG"; }

log "watcher started (PID $$), giving up at $(date -d @$DEADLINE)"

quiet=0
while true; do
    if (( $(date +%s) >= DEADLINE )); then
        log "deadline reached with the sweep still running, plotting anyway"
        break
    fi

    active=$(squeue --me -r -h -o '%j' 2>/dev/null | grep -c '^sweep_')
    if (( active == 0 )); then
        (( ++quiet ))
        log "queue empty ($quiet/$QUIET_CHECKS_NEEDED)"
        (( quiet >= QUIET_CHECKS_NEEDED )) && break
    else
        (( quiet != 0 )) && log "$active tasks back in the queue, resetting"
        quiet=0
    fi
    sleep $POLL_SECONDS
done

# the sweep watcher would otherwise idle until its own deadline
pkill -f 'bash sweep.sh' && log "stopped the sweep watcher"

log "converting pickles"
uv run python -u "$CONVERTER" --source results --dest results_converted >> "$LOG" 2>&1
log "building summary csvs"
uv run python -u -m bofus.summary >> "$LOG" 2>&1
log "plotting"
uv run python -u -m bofus.plots >> "$LOG" 2>&1
log "done, $(find plots -name '*.pdf' | wc -l) pdfs"

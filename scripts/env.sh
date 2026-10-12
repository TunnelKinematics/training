# Shared environment for running Isaac Lab inside Apptainer on the cluster.
# Usage:
#   source scripts/env.sh
#   isaaclab_exec -p <script.py> [args...]     # same args as ./isaaclab.sh

export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/$USER/isaac}
export ISAAC_SIF=${ISAAC_SIF:-$ISAAC_ROOT/sif/isaac-lab_2.3.2.sif}
export REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Isaac Sim caches are large; keep them on /scratch instead of $HOME.
mkdir -p "$ISAAC_ROOT"/{logs,runs,outputs,cache/kit,cache/ov,cache/ov-data,cache/nv}

isaaclab_exec() {
  if [ ! -f "$ISAAC_SIF" ]; then
    echo "Container not found: $ISAAC_SIF (build it from docker/isaaclab.def)" >&2
    return 1
  fi
  apptainer exec --nv --writable-tmpfs --cleanenv \
    --env PYTHONPATH="$REPO_ROOT/training",PYTHONUNBUFFERED=1 \
    -B "$ISAAC_ROOT/cache/kit:/isaac-sim/kit/cache" \
    -B "$ISAAC_ROOT/cache/ov:$HOME/.cache/ov" \
    -B "$ISAAC_ROOT/cache/ov-data:$HOME/.local/share/ov/data" \
    -B "$ISAAC_ROOT/cache/nv:$HOME/.nv" \
    -B "$ISAAC_ROOT/runs:/workspace/isaaclab/logs" \
    -B "$ISAAC_ROOT/outputs:/workspace/isaaclab/outputs" \
    -B "$REPO_ROOT:$REPO_ROOT" \
    --pwd /workspace/isaaclab \
    "$ISAAC_SIF" /workspace/isaaclab/isaaclab.sh "$@"
}

# run_with_timeout <seconds> <cmd...>: like `timeout`, but works with shell functions and kills
# every process the command started (Isaac Sim leaves children behind otherwise).
run_with_timeout() {
  local secs=$1; shift
  export -f isaaclab_exec
  setsid bash -c "$(printf '%q ' "$@")" &
  local pid=$!
  ( sleep "$secs"; kill -TERM -- -"$pid" 2>/dev/null; sleep 15; kill -KILL -- -"$pid" 2>/dev/null ) &
  local watchdog=$!
  wait "$pid"; local rc=$?
  kill "$watchdog" 2>/dev/null
  return $rc
}

#!/bin/bash
# Smoke test: can Isaac Sim / Isaac Lab train headless on this GPU type?
# Usage (from repo root): sbatch --gres=gpu:h100:1 scripts/slurm/gpu_check.sbatch
#SBATCH --job-name=isaac-check
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/isaac/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/env.sh"

echo "== host: $(hostname)  image: $ISAAC_SIF"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 1) Boot Isaac Sim headless. app.close() exits the process, so print first.
isaaclab_exec -p -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, warp
print('torch', torch.__version__, torch.cuda.get_device_name(0), '| warp', warp.__version__, flush=True)
print('CHECK boot: PASS', flush=True)
app.close()
"

# 2) Short PPO run on the stock Go2 rough-terrain task (GPU physics + raycast height scan).
if isaaclab_exec -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 --headless \
    --num_envs 4096 --max_iterations 50; then
  echo "CHECK train: PASS"
else
  echo "CHECK train: FAIL"
fi

# 3) Camera rendering (RGB + depth). The tutorial loops forever, so run it in its
#    own process group, pass once it produces a depth image, then kill the group.
export -f isaaclab_exec
cam_log=$(mktemp)
setsid bash -c "isaaclab_exec -p scripts/tutorials/04_sensors/add_sensors_on_robot.py \
    --headless --enable_cameras --num_envs 1" > "$cam_log" 2>&1 &
cam_pid=$!
for _ in $(seq 60); do  # up to 5 min
  grep -q "Received shape of depth image" "$cam_log" && break
  kill -0 "$cam_pid" 2>/dev/null || break
  sleep 5
done
if grep -q "Received shape of depth image" "$cam_log"; then
  echo "CHECK camera: PASS"
else
  echo "CHECK camera: FAIL"; tail -20 "$cam_log"
fi
kill -TERM -- -"$cam_pid" 2>/dev/null; sleep 10; kill -KILL -- -"$cam_pid" 2>/dev/null
rm -f "$cam_log"

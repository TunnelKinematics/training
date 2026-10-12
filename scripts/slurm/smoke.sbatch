#!/bin/bash
# Fast pre-flight for a robot: asset check + a few PPO iterations of every task.
# Usage (from repo root): sbatch scripts/slurm/smoke.sbatch [robot_name]
#SBATCH --job-name=smoke
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/isaac/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
robot=${1:-go2w}

echo "== check_robot"
run_with_timeout 600 isaaclab_exec -p "$REPO_ROOT/scripts/check_robot.py" --robot "$robot" --headless
for terrain in Flat Rough; do
  task="Bot-$terrain-$robot-v0"
  # Exit codes from Isaac Sim are unreliable, so check that training actually iterated.
  echo "== $task"
  log=$(mktemp)
  run_with_timeout 900 isaaclab_exec -p "$REPO_ROOT/scripts/rl.py" train --task "$task" --headless \
    --num_envs 512 --max_iterations 3 --run_name smoke > "$log" 2>&1
  if grep -q "Learning iteration 2/" "$log" && ! grep -q "Traceback" "$log"; then
    echo "CHECK $task: PASS"
  else
    echo "CHECK $task: FAIL"; grep -A30 "Traceback" "$log" | tail -30
  fi
  rm -f "$log"
done

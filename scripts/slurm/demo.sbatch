#!/bin/bash
# Record scripted-command demo videos (stand, spin, circle, straight + left turn) of a trained policy.
# Usage (from repo root): sbatch scripts/slurm/demo.sbatch <run dir> [play task] [extra demo args...]
#   sbatch scripts/slurm/demo.sbatch $ISAAC_ROOT/runs/rsl_rl/go2w_flat/<run> Bot-Flat-go2w-Play-v0
# Outputs: <run dir>/videos/demo/*.mp4
#SBATCH --job-name=demo
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/isaac/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/env.sh"

run=${1:?usage: sbatch scripts/slurm/demo.sbatch <run dir> [play task] [args...]}
task=${2:-Bot-Flat-go2w-Play-v0}
shift; [ $# -gt 0 ] && shift
run_with_timeout 1500 isaaclab_exec -p "$REPO_ROOT/scripts/demo.py" --task "$task" \
  --policy "$run/exported/policy.pt" --out "$run/videos/demo" --headless --enable_cameras "$@"

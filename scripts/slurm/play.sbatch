#!/bin/bash
# Roll out a trained policy, record a video, and export it to ONNX/TorchScript.
# Usage (from repo root): sbatch scripts/slurm/play.sbatch <play task> [extra play args...]
#   sbatch scripts/slurm/play.sbatch Bot-Rough-go2w-Play-v0              # latest run, latest checkpoint
#   sbatch scripts/slurm/play.sbatch Bot-Rough-go2w-Play-v0 --load_run 2026-10-12_20-00-00
# Outputs: $ISAAC_ROOT/runs/rsl_rl/<robot>_<terrain>/<run>/{videos/play/*.mp4, exported/policy.{onnx,pt}}
#SBATCH --job-name=play
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

task=${1:?usage: sbatch scripts/slurm/play.sbatch <play task> [args...]}
shift
isaaclab_exec -p "$REPO_ROOT/scripts/rl.py" play --task "$task" --headless --video --video_length 1000 "$@"

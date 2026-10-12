#!/bin/bash
# Train a policy with RSL-RL PPO.
# Usage (from repo root): sbatch scripts/slurm/train.sbatch <task> [extra train args...]
#   sbatch scripts/slurm/train.sbatch Bot-Flat-go2w-v0 --max_iterations 300
#SBATCH --job-name=train
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/isaac/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/env.sh"

task=${1:?usage: sbatch scripts/slurm/train.sbatch <task> [args...]}
shift
isaaclab_exec -p "$REPO_ROOT/scripts/rl.py" train --task "$task" --headless "$@"

#!/bin/bash
# Sanity-check a robot asset (loads, stands, rolls).
# Usage (from repo root): sbatch scripts/slurm/check_robot.sbatch [robot_name]
#SBATCH --job-name=check-robot
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/isaac/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/env.sh"

isaaclab_exec -p "$REPO_ROOT/scripts/check_robot.py" --robot "${1:-go2w}" --headless

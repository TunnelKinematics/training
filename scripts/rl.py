"""Run Isaac Lab's stock RSL-RL train/play scripts with our tasks registered.

    isaaclab_exec -p $REPO_ROOT/scripts/rl.py train --task Bot-Flat-Go2W-v0 --headless
    isaaclab_exec -p $REPO_ROOT/scripts/rl.py play  --task Bot-Flat-Go2W-Play-v0 --headless --video

All arguments after train/play are passed through unchanged. Runs land in $ISAAC_ROOT/runs.
"""

import runpy
import sys

import bot.tasks  # noqa: F401  registers Bot-* gym ids (no Isaac Sim imports)

RSL_RL_DIR = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl"

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
    sys.exit("usage: rl.py {train,play} [isaac lab args...]")

script = f"{RSL_RL_DIR}/{sys.argv[1]}.py"
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, RSL_RL_DIR)  # for their local `import cli_args`
runpy.run_path(script, run_name="__main__")

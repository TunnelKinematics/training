"""Robot registry. To add a robot, drop a module in this folder that defines SPEC (a RobotSpec).

Tasks are registered for every robot found here. Importing this package does not import Isaac Sim.
"""

import importlib
import pkgutil

ROBOT_NAMES = sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name != "spec")


def get_robot(name: str):
    return importlib.import_module(f"{__name__}.{name}").SPEC

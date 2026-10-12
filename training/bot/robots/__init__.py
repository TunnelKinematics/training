"""Robot definitions. Each robot module exposes a RobotSpec; register it in ROBOTS."""

from .go2w import GO2W
from .spec import RobotSpec

ROBOTS: dict[str, RobotSpec] = {"go2w": GO2W}

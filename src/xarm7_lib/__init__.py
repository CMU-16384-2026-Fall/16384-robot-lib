"""Python library for driving a UFACTORY xArm7, in simulation or for real.

`Robot` is the usual entry point: it picks the real arm when the `ROBOT_IP`
environment variable is set and the MuJoCo simulation otherwise, so the same
script runs against either. The backends are also importable directly as
`RealXArm7` and `SimulatedXArm7`.
"""

from .api import RobotInterface
from .robot import Robot
from .safety import (
    DEFAULT_BOX,
    DEFAULT_MARGIN,
    SafetyError,
    SafetyGuard,
    Violation,
)
from .xarm7_mujoco import SimulatedXArm7
from .xarm7_real import RealXArm7, XArmError

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BOX",
    "DEFAULT_MARGIN",
    "RealXArm7",
    "Robot",
    "RobotInterface",
    "SafetyError",
    "SafetyGuard",
    "SimulatedXArm7",
    "Violation",
    "XArmError",
    "__version__",
]

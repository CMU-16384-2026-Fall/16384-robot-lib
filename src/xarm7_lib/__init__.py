"""Python library for driving a UFACTORY xArm7, in simulation or for real.

`Robot` is the usual entry point: it picks the real arm when the `ROBOT_IP`
environment variable is set and the MuJoCo simulation otherwise, so the same
script runs against either. The backends are also importable directly as
`RealXArm7` and `SimulatedXArm7`.

Both carry the xArm Gripper on the flange, driven with `grip` — see
`gripper.py`, including what the safety guard does not know about it.
"""

from .api import RobotInterface
from .free_drive import (
    FREE_JOINTS,
    HOME_POSE,
    LOCKED_TOLERANCE,
    Trajectory,
    free_mask,
)
from .gripper import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GRIPPER_STROKE,
)
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
    "FREE_JOINTS",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "GRIPPER_STROKE",
    "HOME_POSE",
    "LOCKED_TOLERANCE",
    "RealXArm7",
    "Robot",
    "RobotInterface",
    "SafetyError",
    "SafetyGuard",
    "SimulatedXArm7",
    "Trajectory",
    "Violation",
    "XArmError",
    "__version__",
    "free_mask",
]

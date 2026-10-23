"""Python library for driving a UFACTORY xArm7, in simulation or for real.

`Robot` is the usual entry point: it picks the real arm when the `ROBOT_IP`
environment variable is set and the MuJoCo simulation otherwise, so the same
script runs against either. The backends are also importable directly as
`RealXArm7` and `SimulatedXArm7`.

`ZedCapture` and `StereoCalibration` cover the ZED stereo camera, imported on
first use so that the simulator needs neither OpenCV nor the native shim.
"""

import importlib

from .api import RobotInterface
from .free_drive import (
    FREE_JOINTS,
    HOME_POSE,
    LOCKED_TOLERANCE,
    Trajectory,
    free_mask,
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

# Imported on first use, so `import xarm7_lib` costs nothing on a machine that
# only runs the simulator: no OpenCV, no native shim. Python routes a name
# missing from the module through `__getattr__` below.
_CAMERA_EXPORTS = {
    "CalibrationError": "calibration",
    "RESOLUTIONS": "calibration",
    "StereoCalibration": "calibration",
    "CameraError": "camera",
    "ZedCapture": "camera",
    "ZedOpenCapture": "camera",
    "ZedDevice": "camera",
    "devices": "camera",
    "query_serial": "camera",
    "ControlsError": "zed_controls",
    "ZedControls": "zed_controls",
}


def __getattr__(name):
    module = _CAMERA_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        imported = importlib.import_module(f".{module}", __name__)
    except ImportError as error:
        raise ImportError(
            f"{name} is part of the camera support, whose dependencies are an "
            "optional extra: install them with `pip install -e \'.[camera]\'`. "
            f"The simulator does not need them. ({error})"
        ) from error
    value = getattr(imported, name)
    globals()[name] = value  # so this only happens once per name
    return value


def __dir__():
    return sorted(__all__)

__all__ = [
    "CalibrationError",
    "CameraError",
    "ControlsError",
    "DEFAULT_BOX",
    "DEFAULT_MARGIN",
    "FREE_JOINTS",
    "HOME_POSE",
    "LOCKED_TOLERANCE",
    "RESOLUTIONS",
    "RealXArm7",
    "Robot",
    "RobotInterface",
    "SafetyError",
    "SafetyGuard",
    "SimulatedXArm7",
    "StereoCalibration",
    "Trajectory",
    "Violation",
    "XArmError",
    "ZedCapture",
    "ZedControls",
    "ZedDevice",
    "ZedOpenCapture",
    "__version__",
    "devices",
    "free_mask",
    "query_serial",
]

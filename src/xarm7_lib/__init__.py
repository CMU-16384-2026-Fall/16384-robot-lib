"""Python library for driving a UFACTORY xArm7, in simulation or for real.

`Robot` is the usual entry point: it picks the real arm when the `ROBOT_IP`
environment variable is set and the MuJoCo simulation otherwise, so the same
script runs against either. The backends are also importable directly as
`RealXArm7` and `SimulatedXArm7`.

Both carry the xArm Gripper on the flange, driven with `grip`; the safety guard
carries it too, so the box and the self-collision check stop the fingertips
rather than the flange. See `gripper.py`.

`ZedOpenCapture` and `StereoCalibration` cover the ZED stereo camera, imported on
first use so that the simulator needs neither OpenCV nor the native shim.

`load_handeye` gives the camera's pose in the robot's base frame, so that a
point the camera measures can be reached for. It is calibrated once per bench
by a TA with `python -m xarm7_lib.calibrate_handeye` and filed inside this
installation, so there is nothing for a student to run or configure. Unlike the
camera itself this costs no OpenCV, and so is imported normally.
"""

import importlib

from .api import RobotInterface
from .free_drive import (
    FREE_JOINTS,
    HOME_POSE,
    LOCKED_TOLERANCE,
    WATCHED_JOINTS,
    Trajectory,
    stop_on_enter,
    watch_mask,
)
from .gripper import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GRIPPER_STROKE,
)
from .handeye import HandEye, HandEyeError, load_handeye
from .visualize import (
    DemoVisualizer,
    Item,
    RobotMirror,
    VisualizerHandle,
    segment_transform,
    visualizer_serial,
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
    "DemoVisualizer",
    "DEFAULT_MARGIN",
    "FREE_JOINTS",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "GRIPPER_STROKE",
    "HOME_POSE",
    "HandEye",
    "HandEyeError",
    "Item",
    "LOCKED_TOLERANCE",
    "RESOLUTIONS",
    "RealXArm7",
    "Robot",
    "RobotInterface",
    "RobotMirror",
    "SafetyError",
    "SafetyGuard",
    "SimulatedXArm7",
    "StereoCalibration",
    "Trajectory",
    "WATCHED_JOINTS",
    "Violation",
    "VisualizerHandle",
    "XArmError",
    "ZedControls",
    "ZedDevice",
    "ZedOpenCapture",
    "__version__",
    "devices",
    "free_mask",
    "load_handeye",
    "query_serial",
    "segment_transform",
    "stop_on_enter",
    "visualizer_serial",
    "watch_mask",
]

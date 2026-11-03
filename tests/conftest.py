"""Stand-ins for the pieces a visualizer draws into, so the tests need no browser."""

import threading

import numpy as np
import pytest

from xarm7_lib import mjscene


class RecordingNode:
    """A meshcat node that writes down what it was asked to do.

    Same four calls the real one offers, and the same `node[name]` descent, so
    the diff under test cannot tell the difference.
    """

    def __init__(self, path="", log=None):
        self.path = path
        self.log = [] if log is None else log

    def __getitem__(self, name):
        child = "/".join(p for p in (self.path, name) if p)
        return RecordingNode(child, self.log)

    def set_object(self, geometry, material=None):
        self.log.append(("set_object", self.path, geometry, material))

    def set_transform(self, matrix=None):
        self.log.append(("set_transform", self.path, np.array(matrix, dtype=float)))

    def set_property(self, key, value):
        self.log.append(("set_property", self.path, key, value))

    def delete(self):
        self.log.append(("delete", self.path))

    # --- reading the log back ---

    def calls(self, kind=None, path=None):
        return [
            c for c in self.log
            if (kind is None or c[0] == kind) and (path is None or c[1] == path)
        ]

    def last_transform(self, path):
        sent = [c[2] for c in self.log if c[0] == "set_transform" and c[1] == path]
        return sent[-1] if sent else None


class RecordingViewer:
    """Enough of `MeshcatVisualizer` for a visualizer to be published into."""

    def __init__(self):
        self.lock = threading.RLock()
        self.viewer = RecordingNode()

    @property
    def log(self):
        return self.viewer.log


class StubRobot:
    """Enough of `Robot` for a `VisualizerHandle`: a viewer and a base frame.

    The base frame carries the 120 mm the real MJCF has, because getting that
    wrong is the failure these tests exist to catch.
    """

    def __init__(self, viewer=None, base_z=0.12):
        self.viewer = viewer if viewer is not None else RecordingViewer()
        self.base_frame = np.eye(4)
        self.base_frame[2, 3] = base_z


@pytest.fixture
def viewer():
    return RecordingViewer()


@pytest.fixture
def robot(viewer):
    return StubRobot(viewer)


@pytest.fixture
def handeye():
    """A synthetic hand-eye: a camera a metre out on -y, looking back along +y."""
    from xarm7_lib.handeye import BoardSpec, HandEye, Intrinsics

    T_base_cam = np.array(
        [
            [1.0, 0.0, 0.0, 0.30],
            [0.0, 0.0, 1.0, -1.00],
            [0.0, -1.0, 0.0, 0.50],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return HandEye(
        T_base_cam=T_base_cam,
        T_flange_board=np.eye(4),
        intrinsics=Intrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0, baseline=0.12),
        board=BoardSpec(squares_x=5, squares_y=7, square_length=0.02, marker_length=0.015),
        serial="test",
    )


@pytest.fixture
def real_base_frame():
    """The actual base frame out of the MJCF, to pin the 120 mm to the model."""
    import mujoco

    model = mjscene.load_arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return mjscene.base_frame(model, data, mjscene.index_arm(model).jid[0])

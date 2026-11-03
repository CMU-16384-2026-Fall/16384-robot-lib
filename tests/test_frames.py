"""The frame chain: camera metres to base metres to the world the picture is in.

The 120 mm between the arm's base frame and the MJCF's world origin is the
thing these tests exist for. Drawn against the wrong one, everything lands
120 mm off the arm and looks like a bad calibration.
"""

import numpy as np
import pytest

import xarm7_lib.handeye as handeye_module
from xarm7_lib.visualize import UNIT_CYLINDER, VisualizerHandle, rotation_to, segment_transform


class Still:
    """A visualizer that draws one sphere at a fixed point in its own frame."""

    def __init__(self, point, frame="base"):
        self.frame = frame
        self.rate = None
        self._point = np.asarray(point, dtype=float)
        self.polls = 0
        import meshcat.geometry as g

        self._geometry = g.Sphere(0.01)
        self._material = g.MeshLambertMaterial(color=0xFFFFFF)

    def poll(self):
        self.polls += 1

    def geometry(self):
        T = np.eye(4)
        T[:3, 3] = self._point
        return {"dot": (self._geometry, self._material, T)}


def step(handle):
    assert handle._poll()
    assert handle._render()


def test_base_frame_carries_the_120mm(robot, viewer):
    handle = VisualizerHandle(robot, Still([0, 0, 0]), "still", start=False)
    step(handle)

    root = viewer.viewer.last_transform("visualizers/still")
    assert root is not None, "the subtree root was never placed"
    np.testing.assert_allclose(root, robot.base_frame)
    assert root[2, 3] == pytest.approx(0.12)


def test_the_120mm_matches_the_model(real_base_frame):
    """Pin the offset to the MJCF, not to a number written in a test."""
    assert real_base_frame[2, 3] == pytest.approx(0.12)
    np.testing.assert_allclose(real_base_frame[:3, :3], np.eye(3), atol=1e-12)


def test_world_frame_is_the_identity(robot, viewer):
    handle = VisualizerHandle(robot, Still([0, 0, 0], frame="world"), "still", start=False)
    step(handle)
    np.testing.assert_allclose(
        viewer.viewer.last_transform("visualizers/still"), np.eye(4)
    )


def test_camera_frame_agrees_with_points_to_base(robot, viewer, handeye):
    """A point drawn in camera metres must land where points_to_base says, plus the 120 mm."""
    point = np.array([0.05, -0.10, 1.20])
    handle = VisualizerHandle(
        robot, Still(point, frame="camera"), "still", handeye=handeye, start=False
    )
    step(handle)

    root = viewer.viewer.last_transform("visualizers/still")
    child = viewer.viewer.last_transform("visualizers/still/dot")
    world = (root @ child)[:3, 3]

    in_base = handeye.points_to_base(point).reshape(3)
    np.testing.assert_allclose(world, in_base + [0, 0, 0.12], atol=1e-12)


def test_camera_frame_without_a_calibration_draws_nothing(robot, viewer, monkeypatch):
    """A missing hand-eye must cost the picture, never the run — and never be faked."""

    def refuse(serial=None):
        raise handeye_module.HandEyeError("no calibration on this bench (test)")

    monkeypatch.setattr(handeye_module, "load_handeye", refuse)

    visualizer = Still([0, 0, 1], frame="camera")
    handle = VisualizerHandle(robot, visualizer, "still", start=False)
    step(handle)
    step(handle)

    assert viewer.log == [], "nothing may be drawn without a calibration"
    assert visualizer.polls == 2, "poll() must keep running so readings stay available"
    assert handle.frame_transform is None


def test_an_unknown_frame_is_refused(robot):
    handle = VisualizerHandle(robot, Still([0, 0, 0], frame="elbow"), "still", start=False)
    assert handle._poll()
    assert not handle._render(), "an unknown frame should disable the visualizer"


def test_rotation_to_is_a_rotation_everywhere():
    axes = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0.3, -0.7, 0.2]]
    rng = np.random.default_rng(0)
    axes += list(rng.normal(size=(20, 3)))
    for axis in axes:
        R = rotation_to(axis)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
        unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        np.testing.assert_allclose(R @ [0, 0, 1], unit, atol=1e-9)


def test_rotation_to_handles_straight_down():
    """A stick pointed at the floor is the axis-angle construction's singularity."""
    R = rotation_to([0, 0, -1])
    assert np.all(np.isfinite(R))
    np.testing.assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize(
    "start,end",
    [
        ([0, 0, 0], [0, 0, 0.5]),
        ([0, 0, 0.5], [0, 0, 0]),
        ([0.1, 0.2, 0.3], [0.1, 0.2, -0.7]),
        ([1, 2, 3], [-1, 0, 2]),
        ([0.2, 0.2, 0.9], [0.2, 0.2, 0.4]),
    ],
)
def test_segment_transform_puts_the_cylinder_ends_on_the_points(start, end):
    """The unit cylinder's two end centres must land on the two given points."""
    M = segment_transform(start, end, 0.0125)
    ends = [(M @ np.array([0.0, s * 0.5, 0.0, 1.0]))[:3] for s in (-1, 1)]
    for point in (np.asarray(start, float), np.asarray(end, float)):
        assert any(np.allclose(point, e, atol=1e-9) for e in ends), (point, ends)


def test_segment_transform_scales_the_radius():
    M = segment_transform([0, 0, 0], [0, 0, 1], 0.02)
    # The cylinder's rim at its own +x is `radius` from the axis.
    rim = (M @ np.array([1.0, 0.0, 0.0, 1.0]))[:3]
    assert np.linalg.norm(rim - [0, 0, 0.5]) == pytest.approx(0.02)


def test_unit_cylinder_is_unit():
    assert UNIT_CYLINDER.height == 1.0
    assert UNIT_CYLINDER.radiusTop == 1.0 and UNIT_CYLINDER.radiusBottom == 1.0

"""The diff, the name check and the exception policy.

The two failures worth most here are silent in the field: geometry republished
every frame (a mesh over the socket thirty times a second, which only shows up
as "meshcat feels laggy"), and a name that escapes the visualizer's subtree and
draws over the arm.
"""

import meshcat.geometry as g
import numpy as np
import pytest

from xarm7_lib.visualize import Item, VisualizerHandle


class Scripted:
    """A visualizer that returns whatever the test puts in `items`."""

    frame = "base"
    rate = None

    def __init__(self, items=None):
        self.items = {} if items is None else items
        self.polls = 0
        self.closed = False
        self.poll_error = None
        self.geometry_error = None

    def poll(self):
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error

    def geometry(self):
        if self.geometry_error is not None:
            raise self.geometry_error
        return self.items

    def close(self):
        self.closed = True


def handle_for(robot, visualizer, name="scripted"):
    return VisualizerHandle(robot, visualizer, name, start=False)


def step(handle):
    return handle._poll() and handle._render()


def pose(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


@pytest.fixture
def shaft():
    return g.Cylinder(1.0, radiusBottom=1.0, radiusTop=0.6)


@pytest.fixture
def red():
    return g.MeshLambertMaterial(color=0xFF2020)


# ----------------------------------------------------------------------
# the diff
# ----------------------------------------------------------------------


def test_cached_geometry_is_published_once_and_then_only_moved(robot, viewer, shaft, red):
    """The whole point of keying the diff on uuid: a moving stick costs one message."""
    visualizer = Scripted()
    handle = handle_for(robot, visualizer)

    for z in (0.1, 0.2, 0.3, 0.4):
        visualizer.items = {"blade": Item(shaft, red, pose([0, 0, z]))}
        step(handle)

    path = "visualizers/scripted/blade"
    assert len(viewer.viewer.calls("set_object", path)) == 1
    assert len(viewer.viewer.calls("set_transform", path)) == 4


def test_an_unchanged_pose_sends_nothing(robot, viewer, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 0.3]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    before = len(viewer.log)
    step(handle)
    step(handle)
    assert len(viewer.log) == before, "a scene that did not change should send nothing"


def test_rebuilt_geometry_republishes_and_eventually_warns(robot, viewer, red, capsys):
    """Building the geometry in geometry() instead of __init__ is the trap."""
    visualizer = Scripted()
    handle = handle_for(robot, visualizer)

    for _ in range(120):
        fresh = g.Cylinder(1.0, radiusBottom=1.0, radiusTop=0.6)
        visualizer.items = {"blade": Item(fresh, red, pose([0, 0, 0.3]))}
        step(handle)

    path = "visualizers/scripted/blade"
    assert len(viewer.viewer.calls("set_object", path)) == 120
    assert "republished" in capsys.readouterr().out


def test_a_name_that_stops_appearing_is_hidden_not_deleted(robot, viewer, shaft, red):
    """This is the stale-detection policy: omit the name, and it leaves the screen."""
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 0.3]))})
    handle = handle_for(robot, visualizer)
    step(handle)

    visualizer.items = {}
    step(handle)

    path = "visualizers/scripted/blade"
    assert ("set_property", path, "visible", False) in viewer.log
    assert viewer.viewer.calls("delete") == [], "hidden, so bringing it back is cheap"


def test_a_name_that_comes_back_is_shown_without_republishing(robot, viewer, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 0.3]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    visualizer.items = {}
    step(handle)
    visualizer.items = {"blade": Item(shaft, red, pose([0, 0, 0.3]))}
    step(handle)

    path = "visualizers/scripted/blade"
    assert len(viewer.viewer.calls("set_object", path)) == 1
    assert ("set_property", path, "visible", True) in viewer.log


def test_hiding_is_only_sent_once(robot, viewer, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 0.3]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    visualizer.items = {}
    step(handle)
    before = len(viewer.log)
    step(handle)
    step(handle)
    assert len(viewer.log) == before


def test_a_bare_geometry_and_a_plain_tuple_both_work(robot, viewer, shaft, red):
    visualizer = Scripted({"a": g.Sphere(0.01), "b": (shaft, red, pose([0, 0, 1]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    assert viewer.viewer.calls("set_object", "visualizers/scripted/a")
    assert viewer.viewer.calls("set_object", "visualizers/scripted/b")


def test_two_visualizers_get_separate_subtrees(robot, viewer, shaft, red):
    for label in ("one", "two"):
        handle = handle_for(robot, Scripted({"blade": Item(shaft, red, pose([0, 0, 1]))}), label)
        step(handle)
    paths = {c[1] for c in viewer.viewer.calls("set_object")}
    assert paths == {"visualizers/one/blade", "visualizers/two/blade"}


# ----------------------------------------------------------------------
# names
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "/mujoco/geom_3", "a//b", ".", "a/../b"])
def test_a_name_escaping_the_subtree_disables_the_visualizer(robot, viewer, name, shaft, capsys):
    """meshcat resets to the scene root on an empty path element, so this would
    not be a mislabelled object — it would be one drawn over the arm."""
    handle = handle_for(robot, Scripted({name: shaft}))
    assert handle._poll()
    assert not handle._render()
    assert viewer.viewer.calls("set_object") == [], "nothing may reach the scene"
    assert "undrawable" in capsys.readouterr().out


# ----------------------------------------------------------------------
# failures
# ----------------------------------------------------------------------


def test_poll_gets_retries_then_gives_up(robot, capsys):
    """Hardware throws transiently; one dropped frame must not end the session."""
    visualizer = Scripted()
    visualizer.poll_error = RuntimeError("dropped frame")
    handle = handle_for(robot, visualizer)

    for _ in range(4):
        assert handle._poll(), "should still be retrying"
    assert not handle._poll(), "should give up on the fifth"
    assert "disabled after 5" in capsys.readouterr().out


def test_a_successful_poll_resets_the_count(robot):
    visualizer = Scripted()
    handle = handle_for(robot, visualizer)

    visualizer.poll_error = RuntimeError("blip")
    for _ in range(4):
        handle._poll()
    assert handle.errors == 4
    visualizer.poll_error = None
    handle._poll()
    assert handle.errors == 0


def test_geometry_failing_disables_immediately(robot, capsys):
    """Unlike poll, this is pure and deterministic: it will fail the same way forever."""
    visualizer = Scripted()
    visualizer.geometry_error = ValueError("bug in the visualizer")
    handle = handle_for(robot, visualizer)

    assert handle._poll()
    assert not handle._render()
    assert "geometry() raised" in capsys.readouterr().out


def test_a_disabled_visualizer_hides_what_it_drew(robot, viewer, shaft, red):
    """Reinforcement for the worst failure here: a stale stick left on screen."""
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 0.3]))})
    handle = handle_for(robot, visualizer)
    step(handle)

    visualizer.poll_error = RuntimeError("camera gone")
    for _ in range(5):
        handle._poll()

    assert ("set_property", "visualizers/scripted/blade", "visible", False) in viewer.log


def test_no_viewer_still_polls(robot, shaft):
    """A dead browser must not stop the data path."""
    robot.viewer = None
    visualizer = Scripted({"blade": shaft})
    handle = handle_for(robot, visualizer)
    for _ in range(3):
        assert step(handle)
    assert visualizer.polls == 3


def test_a_viewer_that_dies_mid_run_is_noticed(robot, viewer, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 1]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    robot.viewer = None  # what Robot.viewer does once the browser goes away
    assert step(handle)
    assert visualizer.polls == 2


# ----------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------


def test_stop_closes_the_visualizer_and_clears_the_subtree(robot, viewer, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 1]))})
    handle = handle_for(robot, visualizer)
    step(handle)
    handle.stop()

    assert visualizer.closed
    assert ("delete", "visualizers/scripted") in viewer.log


def test_stop_is_idempotent(robot, shaft):
    visualizer = Scripted({"blade": shaft})
    handle = handle_for(robot, visualizer)
    handle.stop()
    handle.stop()
    assert visualizer.closed


def test_refresh_frame_reloads_the_transform(robot, viewer, handeye, shaft, red):
    visualizer = Scripted({"blade": Item(shaft, red, pose([0, 0, 1]))})
    visualizer.frame = "camera"
    handle = VisualizerHandle(robot, visualizer, "scripted", handeye=handeye, start=False)
    step(handle)
    first = handle.frame_transform
    assert first is not None

    handle.refresh_frame()
    assert handle.frame_transform is None, "no transform is in force until it is rebuilt"

    step(handle)
    np.testing.assert_allclose(handle.frame_transform, first)
    assert handle._handeye is handeye, "an explicitly given calibration survives a refresh"

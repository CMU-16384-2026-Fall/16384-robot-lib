"""Drawing things beside the arm, and drawing an arm that is somewhere else.

Two halves, both generic — nothing here knows what a light stick is.

`RobotMirror` is the picture of the real arm. The simulator already draws
itself, because it owns a MuJoCo model it is stepping; the real arm owns no
model at all, so this builds one, writes the measured joint angles into it, and
runs forward kinematics only. No physics: a mirror that stepped physics would
droop away from the arm it is supposed to be mirroring.

`register_visualizer` is the other half. A visualizer is any object that can
say what it wants in the scene; it gets its own subtree, its own thread, and a
root transform that carries its coordinates into the world — so an object
working in camera metres hands over the numbers its detector produced and never
touches the hand-eye calibration, the base frame or the 120 mm between them.

The thread is the point. Reading a camera and finding something in the picture
takes 60-140 ms, and meshcat speaks over a REQ/REP socket that has to be held
exclusively while it is written to. Doing both on the arm's 30 Hz clock would
make the arm stutter in time with the camera, so the slow part runs with no
lock held and only the handful of resulting messages are sent under it.
"""

import threading
import time
import traceback
import typing

import meshcat.geometry as g
import mujoco
import numpy as np

from . import mjscene
from .mujoco_meshcat import Y_TO_Z, MeshcatVisualizer

DEFAULT_VISUALIZER_RATE = 30.0  # Hz, matching the simulator's viz_rate
STOP_TIMEOUT = 2.0  # s to wait for a visualizer thread; over one grab + detect
MAX_POLL_ERRORS = 5  # consecutive poll() failures before giving up on an object
REPUBLISH_WARNING = 100  # republishes of one name before we say something

VISUALIZER_PATH = "visualizers"  # alongside "mujoco" and "safety"

# A cylinder of height 1 and radius 1, along +Y, centred on the origin — the
# shape `segment_transform` scales and places. One geometry, published once,
# however much the thing it is drawing moves or changes length.
UNIT_CYLINDER = g.Cylinder(1.0, 1.0)


def rotation_to(axis):
    """A 3x3 rotation taking +z onto `axis`.

    Built as a Householder reflection rather than from an axis-angle, because
    the axis-angle construction divides by the sine of the angle and so blows
    up exactly when `axis` is -z. That is not a corner case here: -z is a
    blade pointed at the floor, which happens in the first session.

    Reflecting +z onto `axis` through the plane bisecting them is a rotation
    composed with a flip, and the flip is undone by reflecting again in the
    plane containing both; the product below is the result of the two, and is
    well conditioned everywhere except the antipode, where the bisecting plane
    is undefined and a half-turn about x is as good an answer as any.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm

    v = axis + np.array([0.0, 0.0, 1.0])
    scale = v @ v
    if scale < 1e-12:  # axis is -z; any half-turn about an axis in xy will do
        return np.diag([1.0, -1.0, -1.0])
    return 2.0 * np.outer(v, v) / scale - np.eye(3)


def segment_transform(start, end, radius=1.0):
    """A 4x4 placing `UNIT_CYLINDER` as the segment from `start` to `end`.

        M = T(midpoint) @ R(+z -> axis) @ Y_TO_Z @ diag(radius, length, radius, 1)

    The order is load-bearing. The scale is rightmost so it acts in the
    cylinder's own frame, where the length is along +Y; `Y_TO_Z` then carries
    that to +Z, which is the axis the rotation expects.

    Length and radius live in the transform rather than in the geometry, which
    is what makes a moving segment cheap: its length changes every frame, and a
    `Cylinder(length, radius)` rebuilt per frame would push a new mesh over the
    socket thirty times a second. This pushes one matrix.
    """
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    along = end - start
    length = float(np.linalg.norm(along))

    scale = np.diag([float(radius), max(length, 1e-9), float(radius), 1.0])
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_to(along)

    placed = rotation @ Y_TO_Z @ scale
    placed[:3, 3] = 0.5 * (start + end)
    return placed


class RobotMirror:
    """A meshcat picture of an arm that is somewhere else.

    Carries no physics. Joint angles are read from `source` — anything with a
    `joint_values` — and written straight into a MuJoCo model whose only job is
    turning seven numbers into geom placements. `mj_kinematics` is the whole of
    it: it is the pass that fills `geom_xpos`/`geom_xmat`, which is all the
    visualizer reads. `mj_forward` would also run collision, passive forces and
    actuation, thirty times a second, for nothing.
    """

    def __init__(self, source, guard=None, mjcf_path=None, gripper=False,
                 viz_rate=DEFAULT_VISUALIZER_RATE, zmq_url=None,
                 open_browser=True, name="xarm7-mirror"):
        self.model = mjscene.load_arm_model(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.index = mjscene.index_arm(self.model)
        if not gripper:
            mjscene.hide_gripper(self.model, mjscene.find_gripper_geoms(self.model))

        self._source = source
        self._guard = guard

        # Before the visualizer, not after: its constructor syncs, and would
        # otherwise paint the arm at all-zeros for one frame. It is also what
        # puts something in `data.xpos` for `base_frame` to read.
        self.set_joints(source.joint_values)

        self.viz = MeshcatVisualizer(
            self.model, self.data, zmq_url=zmq_url, open_browser=open_browser
        )
        mjscene.draw_box(self.viz, guard, self.base_frame)

        self._period = 1.0 / viz_rate if viz_rate else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def base_frame(self):
        """The arm's base frame in the world, as a 4x4 — see `mjscene.base_frame`."""
        return mjscene.base_frame(self.model, self.data, self.index.jid[0])

    @property
    def url(self):
        return None if self.viz is None else self.viz.url

    def set_joints(self, q):
        """Put the arm at `q` and work out where that leaves its geometry."""
        self.data.qpos[self.index.qadr] = np.asarray(q, dtype=float).reshape(7)
        mujoco.mj_kinematics(self.model, self.data)

    def sync(self, force=False):
        """Push the current placement to the browser."""
        if self.viz is None:
            return
        mjscene.update_faces(self.viz, self._guard, self.data.qpos[self.index.qadr])
        self.viz.sync(force=force)

    def close(self):
        """Stop mirroring and drop the viewer."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.viz is not None:
            self.viz.close()
            self.viz = None

    def _run(self):
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self.set_joints(self._source.joint_values)
                self.sync()
            except Exception as err:
                # The same contract as the simulator's `_maybe_sync`: a dead
                # browser, or a dropped connection to the arm, must not take
                # the arm with it. Say so once and stop drawing.
                print(f"[xarm7_lib] mirror disabled: {err!r}")
                self.viz = None
                return
            if self._period:
                elapsed = time.perf_counter() - started
                # wait(), not sleep(), so close() does not have to outwait a
                # whole frame to return.
                self._stop.wait(max(0.0, self._period - elapsed))


class Item(typing.NamedTuple):
    """One named thing in a visualizer's scene.

    Exactly `meshcat.Visualizer.set_object`'s arguments, plus a pose: either a
    geometry and a material, or a meshcat `Object` with `material=None`. The
    protocol's currency is meshcat's own, so a point cloud, a line set or a
    textured mesh goes through unchanged.

    `transform` is read in the frame the visualizer declared, not the world.
    """

    geometry: object
    material: object = None
    transform: object = None  # 4x4, or None for the identity


class Visualizer(typing.Protocol):
    """What `Robot.register_visualizer` wants. Nothing to inherit.

    `poll` does the slow work — read a camera, find something — with no lock
    held, so it may block for as long as it needs to. `geometry` is called
    straight after, must be cheap, and says what should be in the scene *now*:
    a name that stops appearing is hidden, which is how a stale detection
    leaves the picture without anyone having to remember to remove it.

    `draw` is an escape hatch for the things a dict of items cannot say —
    animations, arbitrary properties, a hierarchy built by hand. It is called
    with this visualizer's own meshcat node and the viewer's lock held, so it
    must not block.

    `serial` matters only when `frame` is "camera". A ZED will not say which
    camera it is without being opened, and a visualizer in camera coordinates
    is usually holding that camera open already — so the library asks the
    visualizer for it rather than the bus, and a detector that owns a camera
    should expose it, either as `serial` or on a `camera` attribute. Without it
    the lookup falls back to probing, which fails on exactly the camera being
    drawn from.
    """

    frame: str = "base"  # "base", "camera" or "world"
    rate: float | None = None  # Hz; None means unpaced, poll() is the clock
    serial: str | None = None  # which camera, when `frame` is "camera"

    def poll(self) -> None: ...
    def geometry(self) -> dict: ...  # optional
    def draw(self, node) -> None: ...  # optional
    def close(self) -> None: ...  # optional


def visualizer_serial(visualizer):
    """Which camera `visualizer` is looking through, or None if it cannot say.

    A ZED's serial number can only be read by opening the camera, and a
    visualizer drawing in camera coordinates is usually holding that camera
    open itself -- so going to the bus for it would mean opening a camera that
    is already streaming, which fails, and the calibration keyed on that serial
    would then be unfindable for the one camera we are certainly looking
    through. The object in front of us already knows, so it is asked first: a
    `serial` of its own, or the `serial` of a `camera` it exposes.

    None means fall back to probing, which is the right answer for a visualizer
    that owns no camera.
    """
    for path in (("serial",), ("camera", "serial")):
        value = visualizer
        try:
            for attribute in path:
                value = getattr(value, attribute)
        except Exception:
            # Asking must not be what breaks the drawing: `camera` may be a
            # property, and a property is free to raise. The fallback is fine.
            continue
        if value is not None:
            return str(value)
    return None


def _as_item(value):
    """Accept an `Item`, a plain tuple, or a bare geometry."""
    if isinstance(value, Item):
        return value
    if isinstance(value, tuple):
        return Item(*value)
    return Item(value)


def _check_name(name):
    """Reject a name that would not stay inside the visualizer's own subtree.

    meshcat splits a path on "/" and *resets to the root* on an empty element,
    so "" or "/mujoco/geom_3" from a visualizer would not be a mislabelled
    object, it would be one drawn over the arm. Cheap to check, and the failure
    it prevents is silent.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"visualizer item name must be a non-empty string, got {name!r}")
    if name.startswith("/") or "//" in name:
        raise ValueError(
            f"item name {name!r} would escape the visualizer's subtree: meshcat "
            "resets to the scene root on an empty path element"
        )
    if any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"item name {name!r} must not contain a '.' or '..' element")
    return name


class _Published(object):
    """What we last sent for one name, so the next frame can send only the diff."""

    __slots__ = ("key", "transform", "visible", "republishes")

    def __init__(self, key, transform):
        self.key = key
        self.transform = transform
        self.visible = True
        self.republishes = 0


class VisualizerHandle:
    """One registered visualizer: its thread, its subtree, its frame.

    Returned by `Robot.register_visualizer`. Keep it to stop the visualizer
    again; `Robot.close` stops every one it handed out.
    """

    def __init__(self, robot, visualizer, label, rate=None, handeye=None, start=True):
        self.visualizer = visualizer
        self.name = label
        self.path = f"{VISUALIZER_PATH}/{label}"
        self.errors = 0

        self._robot = robot
        # Told to us, versus found on disk: a refresh re-reads the file but must
        # not throw away a calibration the caller handed over.
        self._given_handeye = handeye
        self._handeye = handeye
        self._handeye_failed = False
        self._frame = None  # the frame string the current transform was built for
        self._frame_transform = None
        self._published = {}
        self._warned_republish = set()
        self._stopped = False

        if rate is None:
            rate = getattr(visualizer, "rate", DEFAULT_VISUALIZER_RATE)
        self._period = 1.0 / rate if rate else None

        self._stop = threading.Event()
        # `start=False` leaves the loop unrun so a caller can step it by hand;
        # the tests drive `_poll`/`_render` directly.
        self._thread = None
        if start:
            self._thread = threading.Thread(
                target=self._run, name=f"xarm7-viz-{label}", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_transform(self):
        """The 4x4 carrying this visualizer's coordinates into the world."""
        return None if self._frame_transform is None else self._frame_transform.copy()

    def refresh_frame(self):
        """Work the frame transform out again.

        Worth calling if the hand-eye calibration has been re-filed while this
        was running — a TA re-running `calibrate_handeye` is the only case that
        comes up in practice, which is why this is on demand and not polled.
        """
        self._handeye = self._given_handeye
        self._handeye_failed = False
        self._frame = None
        self._frame_transform = None

    def stop(self):
        """Stop the thread, let the visualizer go, and clear its drawings.

        The order matters. The thread is joined *before* `close()` is called,
        because `close()` is where a visualizer releases its camera, and
        releasing a camera another thread is mid-read on is a crash inside the
        driver rather than an exception. If the join times out we warn and
        leave the object alone instead.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()

        thread, self._thread = self._thread, None
        joined = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=STOP_TIMEOUT)
            joined = not thread.is_alive()

        if not joined:
            print(
                f"[xarm7_lib] {self.name} did not stop within {STOP_TIMEOUT:g}s; "
                "leaving it running rather than closing what it is using"
            )
        else:
            close = getattr(self.visualizer, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    print(f"[xarm7_lib] {self.name}.close() raised:\n{traceback.format_exc()}")

        self._delete_subtree()

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            started = time.perf_counter()

            if not self._poll():
                break
            if not self._render():
                break

            if self._period:
                self._stop.wait(max(0.0, self._period - (time.perf_counter() - started)))

    def _poll(self):
        """Phase 1: the slow work, with no lock held. False to give up."""
        try:
            self.visualizer.poll()
        except Exception:
            self.errors += 1
            if self.errors == 1:
                print(f"[xarm7_lib] {self.name}.poll() raised:\n{traceback.format_exc()}")
            if self.errors >= MAX_POLL_ERRORS:
                # Hardware throws transiently — a dropped USB frame is not a
                # reason to stay dark for the rest of the session — so this one
                # gets retries. It still gets a limit, so a genuine bug does
                # not print a traceback thirty times a second forever.
                print(
                    f"[xarm7_lib] {self.name} disabled after {self.errors} "
                    "consecutive poll() failures"
                )
                self._hide_all()
                return False
            return True
        self.errors = 0
        return True

    def _render(self):
        """Phases 2 and 3: build the scene, then send what changed."""
        viewer = self._robot.viewer
        if viewer is None:
            return True  # nothing to draw into; keep polling

        geometry = getattr(self.visualizer, "geometry", None)
        draw = getattr(self.visualizer, "draw", None)
        if geometry is None and draw is None:
            return True

        items = None
        if geometry is not None:
            # Deliberately outside the lock: this touches no viewer, so holding
            # the socket across it would only let a slow visualizer stall the
            # arm's own 30 Hz sync.
            try:
                produced = geometry() or {}
            except Exception:
                print(f"[xarm7_lib] {self.name}.geometry() raised:\n{traceback.format_exc()}")
                self._hide_all()
                return False

            try:
                items = {_check_name(name): _as_item(value) for name, value in produced.items()}
            except Exception as err:
                print(f"[xarm7_lib] {self.name} asked for something undrawable: {err}")
                self._hide_all()
                return False

        try:
            with viewer.lock:
                if not self._place_subtree(viewer):
                    return True  # no frame to draw in yet; try again next time
                node = viewer.viewer[self.path]
                if items is not None:
                    self._publish(node, items)
                if draw is not None:
                    draw(node)
        except Exception:
            print(f"[xarm7_lib] {self.name} drawing raised:\n{traceback.format_exc()}")
            return False
        return True

    # ------------------------------------------------------------------
    # the frame, and the diff
    # ------------------------------------------------------------------

    def _place_subtree(self, viewer):
        """Set this visualizer's root transform, so it can draw in its own frame.

        Everything below the root inherits it, which is the whole trick: a
        visualizer working in camera metres hands over the numbers its detector
        produced, and this carries them to the world.
        """
        frame = getattr(self.visualizer, "frame", "base")
        if frame == self._frame and self._frame_transform is not None:
            return True

        transform = self._resolve_frame(frame)
        if transform is None:
            return False

        self._frame = frame
        self._frame_transform = transform
        viewer.viewer[self.path].set_transform(transform)
        # The root moved, so every pose under it is suspect; the easiest
        # correct thing is to let the next frame re-send them.
        for state in self._published.values():
            state.transform = None
        return True

    def _resolve_frame(self, frame):
        if frame == "world":
            return np.eye(4)

        base = np.asarray(self._robot.base_frame, dtype=float)
        if frame == "base":
            return base
        if frame != "camera":
            raise ValueError(
                f"{self.name}.frame is {frame!r}; expected 'base', 'camera' or 'world'"
            )

        handeye = self._load_handeye()
        if handeye is None:
            return None
        return base @ np.asarray(handeye.T_base_cam, dtype=float)

    def _load_handeye(self):
        """The bench's hand-eye calibration, loaded the first time it is wanted.

        Lazily and on this thread on purpose: finding it without a serial opens
        the camera, which needs OpenCV and the native shim, and that must not be
        what makes `register_visualizer` fail on a machine that has neither.

        The serial comes from the visualizer when it can give one -- see
        `visualizer_serial`. Probing for it is the fallback and not the first
        choice, because the camera it would have to open to ask is, for a
        visualizer drawing in camera coordinates, the one already streaming.
        """
        if self._handeye is not None or self._handeye_failed:
            return self._handeye

        from .handeye import HandEyeError, load_handeye

        try:
            self._handeye = load_handeye(serial=visualizer_serial(self.visualizer))
        except HandEyeError as err:
            # Its own message already lists everywhere it looked and says what
            # to run. Drawing camera coordinates as if they were base
            # coordinates would be worse than drawing nothing: it would be a
            # confident picture of the wrong place.
            self._handeye_failed = True
            print(
                f"[xarm7_lib] {self.name} draws in camera coordinates and there is "
                f"no hand-eye calibration, so it will not be drawn:\n  {err}"
            )
        except Exception:
            self._handeye_failed = True
            print(f"[xarm7_lib] {self.name} could not load a hand-eye:\n{traceback.format_exc()}")
        return self._handeye

    def _publish(self, node, items):
        """Send the difference between `items` and what is already on screen."""
        for name, item in items.items():
            key = (
                getattr(item.geometry, "uuid", id(item.geometry)),
                None if item.material is None else getattr(item.material, "uuid", id(item.material)),
            )
            transform = None if item.transform is None else np.asarray(item.transform, dtype=float)
            state = self._published.get(name)

            if state is None or state.key != key:
                node[name].set_object(item.geometry, item.material)
                if state is None:
                    state = self._published[name] = _Published(key, None)
                else:
                    state.key = key
                    state.transform = None
                    state.republishes += 1
                    self._warn_republish(name, state)
                state.visible = True

            if transform is None:
                transform = np.eye(4)
            if state.transform is None or not np.array_equal(state.transform, transform):
                node[name].set_transform(transform)
                state.transform = transform

            if not state.visible:
                node[name].set_property("visible", True)
                state.visible = True

        for name, state in self._published.items():
            if name not in items and state.visible:
                # Hidden rather than deleted: a detection that blinks costs two
                # small messages, where deleting would mean republishing the
                # mesh every time it came back.
                node[name].set_property("visible", False)
                state.visible = False

    def _warn_republish(self, name, state):
        if state.republishes < REPUBLISH_WARNING or name in self._warned_republish:
            return
        self._warned_republish.add(name)
        print(
            f"[xarm7_lib] {self.name} has republished {name!r} "
            f"{state.republishes} times. Build the geometry once in __init__ and "
            "put the pose in the transform; rebuilding it sends a new mesh every frame."
        )

    def _hide_all(self):
        viewer = self._robot.viewer
        if viewer is None:
            return
        try:
            with viewer.lock:
                node = viewer.viewer[self.path]
                for name, state in self._published.items():
                    if state.visible:
                        node[name].set_property("visible", False)
                        state.visible = False
        except Exception:
            pass

    def _delete_subtree(self):
        viewer = self._robot.viewer
        if viewer is None:
            return
        try:
            with viewer.lock:
                viewer.viewer[self.path].delete()
        except Exception:
            pass
        self._published.clear()
        self._frame = None
        self._frame_transform = None


class DemoVisualizer:
    """A stick tumbling around the arm. Needs no camera and no hardware.

    Stands in for a detector: same protocol, same frame, same shape, but the
    pose comes from a clock instead of a picture. It is how the plumbing gets
    tested on a machine with neither a robot nor a camera, and it is the
    shortest example of what a visualizer looks like.

    The fault arguments are for exercising the failure paths on purpose —
    two of them are silent when they go wrong in the field, so they need a
    way to be provoked.
    """

    frame = "base"

    def __init__(self, rate=DEFAULT_VISUALIZER_RATE, length=0.5, radius=0.0125,
                 orbit=0.42, height=0.35, period=6.0, color=0x3FC1C9,
                 raise_in=None, churn=False, bad_name=False):
        self.rate = rate
        self._length = length
        self._radius = radius
        self._orbit = orbit
        self._height = height
        self._period = period
        self._color = color
        self._raise_in = raise_in
        self._churn = churn
        self._bad_name = bad_name
        self._t = 0.0

        # Once, in __init__ — so the diff only ever moves it. Tapered, as the
        # real stick is: `Cylinder` takes two radii.
        self._shaft = g.Cylinder(1.0, radiusBottom=1.0, radiusTop=0.6)
        self._material = g.MeshLambertMaterial(color=color)

    def poll(self):
        if self._raise_in == "poll":
            raise RuntimeError("DemoVisualizer was asked to fail in poll()")
        self._t = time.monotonic()

    def geometry(self):
        if self._raise_in == "geometry":
            raise RuntimeError("DemoVisualizer was asked to fail in geometry()")

        angle = 2.0 * np.pi * self._t / self._period
        centre = np.array(
            [self._orbit * np.cos(angle), self._orbit * np.sin(angle), self._height]
        )
        # Tumbling in the plane containing the orbit radius and z, so it is
        # edge-on and face-on at some point in every revolution — and points
        # straight down once a turn, which is the rotation's awkward case.
        tilt = 2.0 * np.pi * self._t / (0.7 * self._period)
        along = np.array(
            [np.cos(angle) * np.cos(tilt), np.sin(angle) * np.cos(tilt), np.sin(tilt)]
        )
        half = 0.5 * self._length * along

        shaft = (
            g.Cylinder(1.0, radiusBottom=1.0, radiusTop=0.6) if self._churn else self._shaft
        )
        name = "/mujoco/geom_3" if self._bad_name else "stick"
        return {
            name: Item(shaft, self._material, segment_transform(centre - half, centre + half, self._radius))
        }


def main(argv=None):
    """`python -m xarm7_lib.visualize` — the plumbing, with nothing attached."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mirror", action="store_true",
                        help="drive a RobotMirror from a fake arm instead of the simulator")
    parser.add_argument("--rate", type=float, default=DEFAULT_VISUALIZER_RATE,
                        help="visualizer rate in Hz (default %(default)s)")
    parser.add_argument("--raise-in", choices=("poll", "geometry"),
                        help="make the visualizer fail there, to see the policy")
    parser.add_argument("--churn", action="store_true",
                        help="rebuild the geometry every frame, to see the republish warning")
    parser.add_argument("--bad-name", action="store_true",
                        help="try to draw outside the subtree, to see it refused")
    parser.add_argument("--seconds", type=float, help="stop after this long")
    args = parser.parse_args(argv)

    if args.mirror:
        return _main_mirror(args)
    return _main_robot(args)


class _SineArm:
    """Just enough of an arm for the mirror: seven angles that move."""

    def __init__(self):
        self._t0 = time.monotonic()

    @property
    def joint_values(self):
        t = time.monotonic() - self._t0
        return 0.6 * np.sin(0.5 * t + np.arange(7) * 0.7)


def _main_mirror(args):
    from .safety import SafetyGuard

    mirror = RobotMirror(_SineArm(), guard=SafetyGuard())
    print("mirroring a fake arm; Ctrl-C to stop")
    try:
        deadline = None if args.seconds is None else time.monotonic() + args.seconds
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        mirror.close()
    return 0


def _main_robot(args):
    from .robot import Robot

    demo = DemoVisualizer(
        rate=args.rate, raise_in=args.raise_in, churn=args.churn, bad_name=args.bad_name
    )
    with Robot(sim=True) as robot:
        handle = robot.register_visualizer(demo)
        print(f"drawing {handle.name} at {handle.path}; Ctrl-C to stop")
        try:
            deadline = None if args.seconds is None else time.monotonic() + args.seconds
            said = False
            while deadline is None or time.monotonic() < deadline:
                time.sleep(0.2)
                if not handle.active and not said:
                    said = True
                    print(f"{handle.name} stopped; the arm is still going")
                    if args.seconds is None:
                        time.sleep(1.5)
                        break
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

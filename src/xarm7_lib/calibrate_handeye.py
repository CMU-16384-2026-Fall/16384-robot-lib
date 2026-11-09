"""Measure where the ZED is, in the robot's coordinates. A TA runs this once per bench.

    python -m xarm7_lib.calibrate_handeye run --dry-run  # solve, report, write nothing
    python -m xarm7_lib.calibrate_handeye run            # ... and file the result
    python -m xarm7_lib.calibrate_handeye register-tag   # record the table tag for a filed one

The board and the table tag are printed with `python -m xarm7_lib.fiducials`.

A ChArUco board goes on the gripper. The arm takes it through a spread of
poses, the fixed camera watches, and `handeye.solve`/`handeye.refine` recover
two transforms at once: where the camera is in the base frame, and where the
board ended up on the flange. `handeye.py` has the geometry and the arithmetic;
this module is the hardware around it.

The result is filed under `~/.config/xarm7_lib/handeye/`, this account's own
copy, because the account running a calibration is usually not the one that
can write to the installation -- on a lab machine that is `/opt/16384`, owned
by root. The shared copy the whole class reads lives in the installation, and
`run` finishes by printing the one `sudo` line that moves it there, so the
step that needs root is a single command and not a calibration that dies after
the measuring is done. `--system` writes straight into the installation for a
machine where this already runs as someone who can.

One calibration per station, read by every student through `load_handeye()`.
That is the whole reason this is a separate command rather than something a
student script does: it needs the arm to move, it needs the board, it takes a
few minutes, and getting it wrong quietly is easy.

It is a module of its own rather than an `if __name__ == "__main__"` block in
`handeye.py` for the reason `zed_probe.py` gives: the package imports
`handeye`, and `python -m` on a module already in `sys.modules` executes a
second copy of it.


The three ways this goes wrong quietly
--------------------------------------

Each has a check here, because none of them shows up as an error.

**The board printed at the wrong scale.** A print dialog set to "fit to page"
scales the squares by a few percent, and the calibration then comes out wrong
by that same few percent with residuals that look perfect. `fiducials.py`
writes the board as a PDF at exact scale with its spec printed on it, and `run`
says which board it is being told about before it starts.

**The model and the controller disagreeing about the flange.** Forward
kinematics here comes from `xarm7_description`, the same model `safety.py`
uses. If the controller has a TCP offset configured, or the model's zero is not
the controller's zero, every `B(i)` is wrong by a fixed transform and the
calibration absorbs it silently. `run` compares the two before it moves
anything and refuses to continue if they disagree.

**Poses that do not rotate enough.** Hand-eye is solved by rotation: poses that
differ only in position leave the camera's orientation unconstrained, and the
fit is then happy to put the camera somewhere quite wrong. The generated sweep
deliberately rotates the wrist about all three of its axes in turn, and
`leave_one_out` reports the wander that gives it away.
"""

import argparse
import os
import pathlib
import shlex
import sys
import time

import cv2
import numpy as np

from .fiducials import (
    add_board_arguments,
    add_tag_identity_arguments,
    charuco_board,
    spec_from_args,
)
from .handeye import (
    INSTALLED_DIR,
    USER_DIR,
    HandEye,
    HandEyeError,
    Intrinsics,
    Observation,
    find_filed,
    fit_board_pose,
    fit_camera_shift,
    filename,
    leave_one_out,
    pose,
    refine,
    solve,
)
from .tag_recovery import (
    TAG_ID,
    TAG_LENGTH,
    TAG_RESOLUTION,
    record_reference,
)

# The model the forward kinematics comes from, and the two frames taken off it.
# Same description `safety.py` loads, so the arm the guard checks and the arm
# the calibration measures are the same arm.
_DESCRIPTION = "xarm7_description"
_BASE_FRAME = "link_base"
_FLANGE_FRAME = "link_eef"

# How far the model and the controller may disagree about where the flange is
# before `run` refuses to go on. The two compute it completely differently, so
# a few millimetres of disagreement is ordinary; a centimetre means something
# is actually different, and the usual something is a configured TCP offset.
FK_TOLERANCE = 0.010  # m

WINDOW = "calibrate_handeye"
PREVIEW_SCALE = 0.5
# How close the joints have to be to the target before the preview stops
# pumping frames and the controller's own settle test is asked. Looser than the
# controller's tolerance on purpose: this only decides when to stop drawing.
ARRIVAL_TOLERANCE = 0.01  # rad
MOVE_TIMEOUT = 30.0  # s, a backstop on the frame-pumping loop
# How long the arm waits after each pose with its detections still in the
# window. One frame is too brief to tell a good view from a near miss.
DETECTION_HOLD = 0.5  # s

# Automatic exposure, gain and white balance, switched on whenever this opens
# the camera rather than inherited from whatever last had it, so a bench left
# on fixed settings by some other script does not calibrate in the dark.
CAMERA_CONTROLS = {"aec_agc": 1, "auto_white_balance": 1}

# The generated sweep. The wrist joints carry the rotation, because turning the
# wrist re-orients the board without walking it out of the camera's view; the
# others are nudged for translation spread.
WRIST_JOINTS = (4, 5, 6)  # 0-indexed: joint5, joint6, joint7
WRIST_SWING = 0.40  # rad, the excursion given to one wrist joint at a time
WRIST_JITTER = 0.25  # rad, uniform, on the other two
ARM_JITTER = (0.12, 0.10, 0.10, 0.12)  # rad, on joints 1-4
POSE_ATTEMPTS = 20  # candidates tried per pose before giving up on it

DEFAULT_POSES = 18
DEFAULT_FRAMES = 8  # frames averaged per pose
DEFAULT_SETTLE = 1.5  # s after a move, for the arm to stop ringing
DEFAULT_SPEED = 0.25  # rad/s between poses; no hurry, and it keeps the board still

# A corner has to turn up in this fraction of the frames, in both eyes, to be
# used. A corner that flickers is one the detector is unsure about.
FRAME_QUORUM = 0.6
# Below this many corners a pose is not worth keeping: the stereo fit that
# starts the solver off needs a good spread across the board, not three
# corners in one row.
MIN_CORNERS = 8

# What a good calibration looks like on this bench, for the report to say so.
GOOD_RMS_PX = 0.5
GOOD_SPREAD = 0.005  # m of leave-one-out wander

# Under this the corners stop being findable to the precision recovery relies on.
# On renders, 45 mm at 1.5 m is 24 px and fine at 30 degrees above the table, and
# 14 px and three times worse at 20 degrees.
MIN_TAG_SIDE_PX = 20


# ----------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------


class _Kinematics:
    """Where the flange is, for a given set of joint angles.

    Private to this module on purpose. It is not a general-purpose kinematics
    API -- it answers one question, for the one frame the calibration is
    defined against, and a student reaching for forward kinematics should not
    find it here first.
    """

    def __init__(self, description=_DESCRIPTION):
        from robot_descriptions.loaders.pinocchio import load_robot_description

        import pinocchio as pin

        self._pin = pin
        robot = load_robot_description(description)
        self.model = robot.model
        self.data = self.model.createData()
        self.nq = self.model.nq
        for name in (_BASE_FRAME, _FLANGE_FRAME):
            if not self.model.existFrame(name):
                raise HandEyeError(f"{description} has no frame {name!r}")
        self._base = self.model.getFrameId(_BASE_FRAME)
        self._flange = self.model.getFrameId(_FLANGE_FRAME)

    def flange_pose(self, q):
        """`T_base_flange` as a 4x4, for joint angles `q` in radians."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.nq:
            raise HandEyeError(f"expected {self.nq} joint values, got {q.size}")
        self._pin.framesForwardKinematics(self.model, self.data, q)
        # Relative to the base frame rather than to the model's world, so the
        # result means the same thing as the robot's own base frame even if the
        # description ever mounts the arm somewhere other than the origin.
        relative = self.data.oMf[self._base].actInv(self.data.oMf[self._flange])
        return pose(relative.rotation, relative.translation)


def tool_offset(arm):
    """The configured TCP offset's translation, in metres in the flange frame.

    Zero when nothing is configured, which is also what an older controller
    that does not report one looks like.
    """
    offset = arm.arm.tcp_offset
    if offset is None or len(offset) < 3:
        return np.zeros(3)
    return np.asarray(offset[:3], dtype=float) / 1000.0


def check_kinematics(kinematics, arm, tolerance=FK_TOLERANCE):
    """Check the model against the controller's own idea of where the tool is.

    Forward kinematics here is of the bare flange, while the controller reports
    the tool point -- the flange moved by whatever TCP offset is configured. So
    the comparison is made at the tool: the model's flange pose carries the
    offset forward, and that is what the report is matched against.

    A configured offset is not a problem for the calibration. Every flange pose
    the solver is given comes from the model, and where the board really sits
    on the hand is solved as `T_flange_board`, so an offset the controller
    applies to its own reporting never enters the arithmetic. It only has to be
    accounted for here, or this check would reject a bench that is set up
    correctly and ask the TA to break the class's gripper configuration to get
    past it.

    Only the offset's translation is needed. The tool point is
    `p_flange + R_flange @ offset`, which the offset's *rotation* does not move,
    so this makes no assumption about the controller's euler convention.

    Raises `HandEyeError` if the two still disagree, because a constant error in
    every flange pose is absorbed into the answer rather than showing up in the
    residuals.

    Returns the discrepancy in metres.
    """
    reported = arm.arm.position  # [x, y, z, roll, pitch, yaw], mm and radians
    if reported is None or len(reported) < 3:
        raise HandEyeError(
            "the controller did not report a TCP position, so the forward "
            "kinematics cannot be checked against it"
        )
    controller = np.asarray(reported[:3], dtype=float) / 1000.0
    offset = tool_offset(arm)
    flange = kinematics.flange_pose(arm.joint_values)
    model = flange[:3, 3] + flange[:3, :3] @ offset
    error = float(np.linalg.norm(controller - model))

    if error > tolerance:
        world = arm.arm.world_offset
        shifted = world is not None and np.any(
            np.abs(np.asarray(world[:6], dtype=float)) > 1e-6
        )
        culprit = (
            f"\nThe controller has a base coordinate offset configured "
            f"({world}), which is almost certainly it: it moves the frame the "
            "controller reports in away from the model's `link_base`, and "
            "unlike the tool offset it cannot be read off the report. Clear it "
            "and run this again."
            if shifted
            else "\nThe TCP offset above is already accounted for, so that is "
            "not it. Check that the controller is really an xArm7 and that its "
            "joint zero matches the model's."
        )
        raise HandEyeError(
            f"forward kinematics and the controller disagree about where the "
            f"tool is by {error * 1000:.1f} mm, which is more than the "
            f"{tolerance * 1000:.0f} mm allowed.\n"
            f"  controller reports  {np.round(controller, 4)} m\n"
            f"  the model computes  {np.round(model, 4)} m\n"
            f"  flange              {np.round(flange[:3, 3], 4)} m\n"
            f"  tool offset         {np.round(offset, 4)} m (in the flange frame)\n"
            "Every flange pose the solver gets comes from the model, so a "
            f"constant error here is absorbed into the calibration instead of "
            f"showing up in its residuals.{culprit}"
        )
    return error


# ----------------------------------------------------------------------
# Seeing the board
# ----------------------------------------------------------------------


class BoardDetector:
    """Finds the board's interior corners in an image.

    Reports `(ids, corners)` with ids being ChArUco corner ids, which index
    `BoardSpec.object_points()` directly. That is what makes the left/right
    correspondence exact: two eyes agree on a corner because they agree on its
    id, not because something matched them geometrically.
    """

    def __init__(self, spec):
        self.spec = spec
        self.board = charuco_board(spec)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.object_points = spec.object_points()

    def detect(self, image):
        """`(ids, corners)` for one image, as `(N,)` ints and `(N, 2)` pixels."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) == 0:
            return np.empty(0, dtype=int), np.empty((0, 2))
        return (
            np.asarray(ids, dtype=int).reshape(-1),
            np.asarray(corners, dtype=float).reshape(-1, 2),
        )


def _average_detections(detector, frames):
    """Average each corner over several frames, per eye.

    Returns `{id: (u, v)}` per eye. Averaging is worth the frames it costs: the
    detector's sub-pixel refinement still scatters by a few tenths of a pixel
    frame to frame, and that scatter is what the final accuracy is made of.
    """
    seen = ({}, {})
    for left, right in frames:
        for eye, image in enumerate((left, right)):
            ids, corners = detector.detect(image)
            for corner_id, corner in zip(ids, corners):
                seen[eye].setdefault(int(corner_id), []).append(corner)

    quorum = max(1, int(round(FRAME_QUORUM * len(frames))))
    return tuple(
        {
            corner_id: np.mean(points, axis=0)
            for corner_id, points in eye.items()
            if len(points) >= quorum
        }
        for eye in seen
    )


def observe(camera, detector, flange, frames=DEFAULT_FRAMES, preview=None, note=()):
    """Look at the board from where the arm is now. Returns an `Observation` or None.

    None, with a reason printed, when the board is not usably in view -- which
    is an ordinary outcome for a generated pose, not an error.

    With a `preview`, the averaged detections are drawn on the last frame
    grabbed, including the ones only one eye saw. A pose that fails is drawn
    too, and before it is reported: the picture of a near miss is the useful
    part, and it is gone by the time the arm has moved on. The window then
    holds that picture for `DETECTION_HOLD` before the arm goes anywhere.
    """
    grabbed = [camera.read_rectified() for _ in range(frames)]
    left, right = _average_detections(detector, grabbed)

    shared = sorted(set(left) & set(right))
    if preview is not None:
        lonely = sorted(set(left) - set(right))
        preview.show(
            grabbed[-1][0],
            corners=[left[i] for i in shared],
            extra=[left[i] for i in lonely],
            note=list(note) + [
                f"{len(shared)} corners in both eyes"
                + (f", {len(lonely)} in the left only" if lonely else "")
            ],
        )
        preview.hold(DETECTION_HOLD)
    if len(shared) < MIN_CORNERS:
        print(
            f"      skipped: {len(shared)} corners in both eyes "
            f"({len(left)} left, {len(right)} right), need {MIN_CORNERS}"
        )
        return None

    ids = np.asarray(shared, dtype=int)
    return Observation(
        flange=flange,
        ids=ids,
        left=np.array([left[i] for i in shared]),
        right=np.array([right[i] for i in shared]),
    )


# ----------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------


class Preview:
    """A window showing the left eye while the routine runs.

    Worth having because the one thing that decides whether this works is
    whether the board is in view and detectable at every pose, and without a
    picture the only evidence of that is a corner count after the fact. With
    one, a board that is edge-on, half out of frame, blown out or still
    blurring from the move is obvious while there is time to do something
    about it.

    Opening it is best-effort. OpenCV built without GUI support, or a session
    with no display, raises on the first call, and a calibration that is
    otherwise fine should not die for want of a window: the reason is printed
    once and the routine carries on blind.
    """

    def __init__(self, size, enabled=True, scale=PREVIEW_SCALE):
        self.scale = scale
        self.enabled = bool(enabled)
        self.key = -1
        if not self.enabled:
            return
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, int(scale * size[0]), int(scale * size[1]))
        except cv2.error as error:
            self._give_up(error)

    def _give_up(self, error):
        self.enabled = False
        reason = str(error).strip().splitlines()[-1] if str(error).strip() else error
        print(f"\n  no preview window ({reason}); carrying on without one\n")

    def show(self, image, corners=None, extra=None, note=()):
        """Draw one frame. Returns the key pressed in the window, or -1.

        `corners` are the detections the solver will use, and `extra` the ones
        it cannot (seen in one eye only).

        Everything is drawn *after* the frame is scaled down, so the markers
        and the caption stay the same size on screen whatever
        `--display-scale` is set to. Drawn before it, a 3-pixel dot at half
        scale is a pixel and a half of nothing.
        """
        self.key = -1
        if not self.enabled:
            return self.key
        canvas = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)
        for points, colour in ((extra, (0, 180, 255)), (corners, (0, 255, 0))):
            if points is None:
                continue
            for u, v in np.asarray(points, dtype=float).reshape(-1, 2) * self.scale:
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                cv2.circle(canvas, (int(round(u)), int(round(v))), 3,
                           colour, -1, cv2.LINE_AA)
        self._caption(canvas, note)
        try:
            cv2.imshow(WINDOW, canvas)
            self.key = self._key(cv2.waitKey(1))
        except cv2.error as error:
            self._give_up(error)
        return self.key

    @staticmethod
    def _key(value):
        """`waitKey`'s answer as a plain key code, or -1 for nothing pressed."""
        return -1 if value == -1 else value & 0xFF

    @staticmethod
    def _caption(canvas, lines):
        """Text at the top left, over a dark strip so it stays readable."""
        lines = [line for line in lines if line]
        if not lines:
            return
        # Sized off the window, not the sensor: the point is on-screen
        # legibility, and the two are the same thing only at scale 1.
        k = min(1.5, max(0.5, canvas.shape[1] / 1280.0))
        step = int(18 + 10 * k)
        height = min(canvas.shape[0], step * len(lines) + step // 2)
        canvas[:height] = (canvas[:height].astype(np.float32) * 0.35).astype(np.uint8)
        for index, line in enumerate(lines, start=1):
            cv2.putText(canvas, line, (int(12 * k), index * step),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 + 0.25 * k,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def hold(self, seconds):
        """Keep the last frame up, and the window responsive, for a while.

        `waitKey` is the GUI's own event pump, so one call is both the wait and
        the thing that stops the window going grey. With no window there is
        nothing to look at, so it returns at once. Returns the key pressed.
        """
        self.key = -1
        if not self.enabled:
            return self.key
        try:
            self.key = self._key(cv2.waitKey(max(1, int(seconds * 1000))))
        except cv2.error as error:
            self._give_up(error)
        return self.key

    @property
    def closed(self):
        """True once the window has been closed from the window manager."""
        if not self.enabled:
            return False
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        try:
            cv2.destroyWindow(WINDOW)
            # A destroyed window is only really gone once the GUI loop runs.
            cv2.waitKey(1)
        except cv2.error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False


# ----------------------------------------------------------------------
# Where to put the arm
# ----------------------------------------------------------------------


def generate_poses(seed, count, check=None, rng=None):
    """A spread of joint configurations around `seed`, all of them allowed.

    Rotation is what solves hand-eye, so each pose gives one wrist joint a
    deliberate swing and the other two a jitter, cycling which joint gets the
    swing. Poses that differ only in position leave the camera's orientation
    unconstrained no matter how many of them there are, and cycling the axis is
    what stops a random sweep from accidentally rotating about mostly one.

    The seed is wherever the arm already is: the TA has put the gripper where
    the camera can see the board, and everything here stays near it.

    Args:
        seed: the joint configuration to work around.
        count: how many poses to return.
        check: `robot.check_safety` or anything with its signature. A candidate
            it refuses is discarded and another is drawn, so the sweep never
            commands something the arm would reject.
        rng: for repeatability.

    Returns:
        A list of `(7,)` arrays, shorter than `count` if the seed is too
        hemmed in to find that many.
    """
    seed = np.asarray(seed, dtype=float).reshape(-1)
    rng = np.random.default_rng() if rng is None else rng

    poses = []
    for index in range(count):
        swing_joint = WRIST_JOINTS[index % len(WRIST_JOINTS)]
        # Alternate the direction, so cycling the axis doesn't also mean always
        # leaning the same way round it.
        direction = 1.0 if (index // len(WRIST_JOINTS)) % 2 == 0 else -1.0

        for _ in range(POSE_ATTEMPTS):
            candidate = seed.copy()
            candidate[swing_joint] += direction * rng.uniform(
                0.5 * WRIST_SWING, WRIST_SWING
            )
            for joint in WRIST_JOINTS:
                if joint != swing_joint:
                    candidate[joint] += rng.uniform(-WRIST_JITTER, WRIST_JITTER)
            for joint, amplitude in enumerate(ARM_JITTER):
                candidate[joint] += rng.uniform(-amplitude, amplitude)

            if check is not None and check(candidate) is not None:
                continue
            poses.append(candidate)
            break

    return poses


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _report(T_base_cam, T_flange_board, stats, spread=None):
    """Print what was solved and how much to believe it."""
    position = T_base_cam[:3, 3]
    forward = T_base_cam[:3, 2]

    print("\n  camera, in the robot's base frame")
    print(
        f"    position     {position[0]:+.4f} {position[1]:+.4f} "
        f"{position[2]:+.4f}  m"
    )
    print(
        f"    looks along  {forward[0]:+.4f} {forward[1]:+.4f} {forward[2]:+.4f}"
    )
    print(f"    distance from base origin   {np.linalg.norm(position):.3f} m")

    board = T_flange_board[:3, 3]
    print("\n  board, on the flange")
    print(f"    offset       {board[0]:+.4f} {board[1]:+.4f} {board[2]:+.4f}  m")

    print("\n  fit")
    print(f"    poses used                  {stats['poses']}")
    print(f"    corners used                {stats['corners']}")
    print(
        f"    reprojection RMS            {stats['rms_px']:.3f} px"
        f"   (before refining: {stats['rms_px_before_refine']:.3f})"
    )
    print(f"    worst single corner         {stats['max_px']:.3f} px")
    worst = int(np.argmax(stats["per_pose_rms_px"]))
    print(
        f"    worst pose                  #{worst + 1} at "
        f"{stats['per_pose_rms_px'][worst]:.3f} px"
    )
    if spread is not None:
        print(f"    leave-one-out wander        {spread * 1000:.2f} mm")

    print()
    if stats["rms_px"] > GOOD_RMS_PX:
        print(
            f"  ! reprojection RMS is above {GOOD_RMS_PX} px. Something is "
            "mis-detected, or the\n"
            "    board is not the size it was declared to be. Check the worst "
            "pose above."
        )
    elif spread is not None and spread > GOOD_SPREAD:
        print(
            f"  ! the camera moves by {spread * 1000:.0f} mm when a single pose "
            "is dropped, which is\n"
            "    more than the residuals suggest. The poses probably do not "
            "rotate enough —\n"
            "    collect more, turning the wrist further between them."
        )
    else:
        print("  this looks good: the corners fit and no single pose is holding it up.")


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def _require_ip(args):
    """Fail before anything slow happens if there is no arm to talk to.

    Called first, because loading the collision meshes takes a few seconds and
    there is no point spending them to discover that ROBOT_IP is unset.
    """
    if not args.ip:
        raise HandEyeError(
            "no controller address: pass --ip, or set ROBOT_IP. This needs the "
            "real arm — there is nothing to calibrate in simulation."
        )


def _open_arm(args, guard=True):
    """Connect to the real arm. `guard=False` also skips loading the meshes."""
    from .xarm7_real import RealXArm7

    _require_ip(args)
    return RealXArm7(args.ip, guard=guard and not getattr(args, "no_guard", False))


def _open_camera(args):
    from .camera import ZedOpenCapture

    return ZedOpenCapture(
        resolution=args.resolution, serial=args.serial, controls=CAMERA_CONTROLS
    )


def _stream(camera, preview, seconds, note=()):
    """Hold the window open for a while with nothing to decide."""
    if not preview.enabled:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        preview.show(camera.read_rectified()[0], note=note)


def _move_watching(arm, camera, preview, goal, speed, note=()):
    """Move to `goal`, keeping frames coming while the arm travels.

    `set_joint_targets(wait=True)` blocks, and a frozen window beside a moving
    robot reads as a hung program. So with a window the move is started without
    waiting and the preview is pumped until the joints stop changing, after
    which the controller's own settle test has the last word exactly as it does
    when the call is left to block. With no window this *is* that call.
    """
    if not preview.enabled:
        arm.set_joint_targets(goal, speed=speed, wait=True)
        return
    arm.set_joint_targets(goal, speed=speed, wait=False)
    deadline = time.monotonic() + MOVE_TIMEOUT
    while time.monotonic() < deadline:
        preview.show(camera.read_rectified()[0], note=list(note) + ["moving"])
        if np.max(np.abs(arm.joint_values - goal)) < ARRIVAL_TOLERANCE:
            break
    arm.wait_for_motion()


def _capture_automatic(arm, camera, detector, kinematics, args, preview):
    """Walk the arm through a generated sweep, observing at each stop."""
    seed = arm.joint_values
    print(f"  seed pose {np.round(np.degrees(seed), 1)} deg (where the arm is now)")

    poses = generate_poses(
        seed,
        args.poses,
        check=None if args.no_guard else arm.check_safety,
        rng=np.random.default_rng(args.seed),
    )
    if len(poses) < args.poses:
        print(
            f"  only {len(poses)} of {args.poses} generated poses are allowed "
            "from here — the seed pose is close to a limit or the safety box"
        )
    if not poses:
        raise HandEyeError(
            "no allowed poses near the seed pose. Move the arm somewhere with "
            "more room and try again."
        )

    observations = []
    try:
        for index, q in enumerate(poses, start=1):
            print(f"\n  pose {index}/{len(poses)}")
            label = f"pose {index}/{len(poses)}"
            _move_watching(arm, camera, preview, q, args.speed, note=[label])
            _stream(camera, preview, args.settle, note=[label, "settling"])
            # Where the arm actually is, not where it was sent. The two differ
            # by the servos' own steady-state error, and the solver should be
            # told the truth.
            measured = arm.joint_values
            observation = observe(
                camera, detector, kinematics.flange_pose(measured), args.frames,
                preview=preview, note=[label],
            )
            _debug_pose(index, arm, camera, detector, kinematics, q, measured,
                        observation)  # DEBUG (temporary)
            if observation is not None:
                print(f"      {len(observation)} corners")
                observations.append(observation)
    except KeyboardInterrupt:
        print("\n  interrupted — solving with what was collected so far")
        arm.stop()

    return observations


def _pose_from_window(camera, detector, preview, index, frames):
    """Stream, with live detections, until the TA asks for a capture.

    Returns "capture" or "done". The window is the better prompt here: posing
    the arm by hand is a hunt for a view where the board reads, and the corner
    count has to be visible *while* the hunting happens, not after.
    """
    while True:
        left, _ = camera.read_rectified()
        ids, corners = detector.detect(left)
        key = preview.show(
            left,
            corners=corners,
            note=[
                f"pose {index}: space to capture, q when you have enough",
                f"{len(ids)} corners in the left eye"
                + ("" if len(ids) >= MIN_CORNERS else f" -- need {MIN_CORNERS}"),
            ],
        )
        if key in (ord(" "), 13, 10):
            return "capture"
        if key in (ord("q"), 27):
            return "done"
        if preview.closed:
            print("  window closed — finishing capture")
            return "done"


def _capture_manual(arm, camera, detector, kinematics, args, preview):
    """Let the TA push the arm to each pose by hand.

    The controller's teaching mode, the same one `python -m
    xarm7_lib.free_drive` uses, released between captures and locked for them.
    For a bench where the generated sweep puts the board out of the camera's
    view.
    """
    from .xarm7_real import MODE_POSITION, MODE_TEACH

    print(
        "\n  Manual capture. The arm is released between poses — the controller "
        "holds it up\n  against gravity but nothing is watching where it goes. "
        "Keep the e-stop in reach.\n"
        "  Turn the wrist a long way between poses, not just move the arm "
        "around: rotation\n  is what solves this.\n"
        + (
            "  Space in the window captures where it stands, q finishes."
            if preview.enabled
            else "  Press Enter to capture where it stands, or Ctrl-C when you "
            "have enough."
        )
    )

    observations = []
    try:
        while len(observations) < args.poses:
            index = len(observations) + 1
            arm._ensure_mode(MODE_TEACH)
            if preview.enabled:
                if _pose_from_window(
                    camera, detector, preview, index, args.frames
                ) == "done":
                    break
            else:
                try:
                    input(f"\n  pose {index}: push the arm, then Enter: ")
                except EOFError:
                    print("  (nothing on stdin to wait for)")
                    break
            arm._ensure_mode(MODE_POSITION)
            _stream(camera, preview, args.settle,
                    note=[f"pose {index}", "settling"])

            measured = arm.joint_values
            observation = observe(
                camera, detector, kinematics.flange_pose(measured), args.frames,
                preview=preview, note=[f"pose {index}"],
            )
            _debug_pose(index, arm, camera, detector, kinematics, None, measured,
                        observation)  # DEBUG (temporary)
            if observation is not None:
                print(f"      {len(observation)} corners")
                observations.append(observation)
    except KeyboardInterrupt:
        print()
    finally:
        # Locked rather than left free: the next thing to touch this arm should
        # find it holding position, as every other path through this library
        # leaves it.
        arm._ensure_mode(MODE_POSITION)

    return observations


def _destination(args, serial):
    """Where the result goes.

    Under `USER_DIR` unless told otherwise, because the account running this is
    usually not the one that can write to the installation: on a lab machine
    that is `/opt/16384`, owned by root. Writing there is a separate step with
    `sudo`, and `cmd_run` prints the command for it rather than failing
    halfway through a calibration that took a few minutes to measure.
    """
    if args.output:
        return pathlib.Path(args.output)
    if args.system:
        return INSTALLED_DIR / filename(serial)
    return USER_DIR / filename(serial)


def _promote_command(written, serial):
    """The `sudo` line that moves a user-filed calibration into the installation.

    Printed, not run: this process is not the one with the access, and a TA
    pasting it sees exactly what is about to happen to the shared copy.
    """
    target = INSTALLED_DIR / filename(serial)
    src = shlex.quote(str(written))
    directory = shlex.quote(str(INSTALLED_DIR))
    dst = shlex.quote(str(target))
    return (
        f"sudo mkdir -p {directory} && sudo mv {src} {dst} && "
        f"sudo chmod 644 {dst}"
    )


# ----------------------------------------------------------------------
# DEBUG (temporary): records everything `run` sees, to chase the
# "points are at or behind the camera plane" failure. Remove once fixed.
# ----------------------------------------------------------------------


_DEBUG = None


class _DebugLog:
    """Writes `handeye_debug/<timestamp>/` in the working directory.

    meta.json        camera calibration, rectification, intrinsics, arm offsets
    pose_NN.npz      joints, FK flange, controller TCP, corners, stereo board fit
    pose_NN_*.png    rectified left/right of one frame grabbed right after observing
    pose_NN_raw_*.jpg  the same frame unrectified
    solve.npz        board poses, flange poses, closed-form guess, per-pose checks
    traceback.txt    if solving failed
    """

    def __init__(self, root="handeye_debug"):
        self.dir = pathlib.Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        print(f"  DEBUG: recording to {self.dir.resolve()}")

    def _json(self, name, data):
        import json

        (self.dir / name).write_text(json.dumps(data, indent=2, default=_debug_plain))

    def meta(self, args, spec, arm, camera, intrinsics):
        calibration = camera.calibration
        R1, R2, P1, P2, Q = calibration.rectify()
        try:
            profile = camera._session.profile()
        except Exception as error:  # noqa: BLE001 -- debug only
            profile = repr(error)
        self._json("meta.json", {
            "args": vars(args),
            "board": spec.to_dict() if hasattr(spec, "to_dict") else repr(spec),
            "object_points": spec.object_points(),
            "camera": {
                "serial": camera.serial,
                "resolution": camera.resolution,
                "size": camera.size,
                "calibration_image_size": calibration.image_size,
                "fisheye": calibration.fisheye,
                "baseline": calibration.baseline,
                "K1": calibration.K1, "D1": calibration.D1,
                "K2": calibration.K2, "D2": calibration.D2,
                "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
                "profile": profile,
            },
            "intrinsics": intrinsics.to_dict(),
            "arm": {
                "joint_values": arm.joint_values,
                "position": arm.arm.position,
                "tcp_offset": arm.arm.tcp_offset,
                "world_offset": arm.arm.world_offset,
            },
        })

    def pose(self, index, arm, camera, detector, intrinsics, kinematics,
             commanded, measured, observation):
        flange = kinematics.flange_pose(measured)
        controller = np.asarray(arm.arm.position, dtype=float)
        offset = tool_offset(arm)
        tool_model = flange[:3, 3] + flange[:3, :3] @ offset
        fk_error = float(np.linalg.norm(controller[:3] / 1000.0 - tool_model))

        raw_left, raw_right = camera.read()
        rect_left, rect_right = camera.calibration.rectify_images(raw_left, raw_right)
        tag = f"pose_{index:02d}"
        cv2.imwrite(str(self.dir / f"{tag}_left.png"), rect_left)
        cv2.imwrite(str(self.dir / f"{tag}_right.png"), rect_right)
        cv2.imwrite(str(self.dir / f"{tag}_raw_left.jpg"), raw_left,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.dir / f"{tag}_raw_right.jpg"), raw_right,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        record = dict(
            commanded=np.asarray(commanded if commanded is not None else measured, float),
            measured=np.asarray(measured, float),
            flange=flange,
            controller_position=controller,
            tcp_offset=offset,
            fk_error_m=fk_error,
        )
        line = f"      DEBUG fk-vs-controller {fk_error * 1000:.1f} mm"
        if observation is not None:
            measured_points = intrinsics.triangulate(observation.left, observation.right)
            known = detector.object_points[observation.ids]
            board = fit_board_pose(observation, intrinsics, detector.object_points)
            fitted = known @ board[:3, :3].T + board[:3, 3]
            geometry_rms = float(np.sqrt(np.mean(np.sum((measured_points - fitted) ** 2, 1))))
            # Mean pairwise distance, measured over known: 1.0 if the stereo scale
            # (baseline, focal length, printed size) is right.
            def spread(p):
                return float(np.mean(np.linalg.norm(p[:, None] - p[None], axis=2)))
            scale = spread(measured_points) / spread(known)
            record.update(
                ids=observation.ids, left=observation.left, right=observation.right,
                triangulated=measured_points, T_cam_board=board,
                geometry_rms_m=geometry_rms, scale_ratio=scale,
                row_error_px=observation.left[:, 1] - observation.right[:, 1],
            )
            line += (
                f", board at z={board[2, 3]:.3f} m, geometry rms "
                f"{geometry_rms * 1000:.2f} mm, scale {scale:.4f}, row error "
                f"{np.max(np.abs(record['row_error_px'])):.2f} px"
            )
        np.savez(self.dir / f"{tag}.npz", **record)
        print(line)

    def solve(self, observations, board_poses, guess_cam, guess_board,
              intrinsics, object_points):
        from .handeye import invert, rotation_angle

        flanges = np.array([o.flange for o in observations])
        boards = np.array(board_poses)
        # Rotation angle is invariant under conjugation, so for every pair of
        # poses the board's relative rotation seen by the camera must equal the
        # flange's relative rotation from FK. A mismatch here is the data, not
        # the solver.
        pairs = []
        for i in range(len(boards)):
            for j in range(i + 1, len(boards)):
                cam = rotation_angle((boards[i] @ invert(boards[j]))[:3, :3], np.eye(3))
                base = rotation_angle((flanges[i] @ invert(flanges[j]))[:3, :3], np.eye(3))
                pairs.append((i, j, np.degrees(cam), np.degrees(base)))
        pairs = np.array(pairs)
        mismatch = np.abs(pairs[:, 2] - pairs[:, 3])
        # Same check for the other convention, in case B or A is inverted.
        swapped = []
        for i in range(len(boards)):
            for j in range(i + 1, len(boards)):
                cam = rotation_angle((invert(boards[i]) @ boards[j])[:3, :3], np.eye(3))
                base = rotation_angle((invert(flanges[i]) @ flanges[j])[:3, :3], np.eye(3))
                swapped.append(abs(np.degrees(cam) - np.degrees(base)))
        print(
            f"  DEBUG pairwise rotation, camera vs FK: median {np.median(mismatch):.3f} deg, "
            f"max {np.max(mismatch):.3f} deg over {len(pairs)} pairs "
            f"(other convention: median {np.median(swapped):.3f} deg)"
        )

        T_cam_base = invert(guess_cam)
        depth = []
        print("  DEBUG closed-form guess vs each measured board pose:")
        for k, (observation, measured) in enumerate(zip(observations, board_poses)):
            predicted = T_cam_base @ observation.flange @ guess_board
            corners = object_points[observation.ids] @ predicted[:3, :3].T + predicted[:3, 3]
            depth.append(float(corners[:, 2].min()))
            angle = np.degrees(rotation_angle(predicted[:3, :3], measured[:3, :3]))
            offset = np.linalg.norm(predicted[:3, 3] - measured[:3, 3])
            print(
                f"    pose {k + 1:2d}: measured z {measured[2, 3]:+.3f} m, predicted z "
                f"{predicted[2, 3]:+.3f} m, min corner z {depth[-1]:+.3f} m, off by "
                f"{angle:.1f} deg / {offset * 1000:.0f} mm"
            )
        np.savez(
            self.dir / "solve.npz",
            flange_poses=flanges, board_poses=boards,
            guess_T_base_cam=guess_cam, guess_T_flange_board=guess_board,
            pair_rotation_deg=pairs, pair_rotation_deg_swapped_mismatch=np.array(swapped),
            predicted_min_depth=np.array(depth),
            ids=np.array([o.ids for o in observations], dtype=object),
            left=np.array([o.left for o in observations], dtype=object),
            right=np.array([o.right for o in observations], dtype=object),
        )

    def failure(self):
        import traceback

        (self.dir / "traceback.txt").write_text(traceback.format_exc())
        print(f"  DEBUG: traceback written to {self.dir / 'traceback.txt'}")


def _debug_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (tuple, set)):
        return list(value)
    return repr(value)


def _debug_pose(index, arm, camera, detector, kinematics, commanded, measured,
                observation):
    if _DEBUG is None:
        return
    try:
        _DEBUG.pose(index, arm, camera, detector, _DEBUG.intrinsics, kinematics,
                    commanded, measured, observation)
    except Exception as error:  # noqa: BLE001 -- recording must not cost the run
        print(f"      DEBUG: could not record pose {index}: {error!r}")


# ---------------------------------------------------------------- end DEBUG


def cmd_run(args):
    _require_ip(args)
    spec = spec_from_args(args)
    detector = BoardDetector(spec)

    print(f"board: {spec.describe()}")
    print("  (if that is not the board on the gripper, stop now)")

    print("\nloading the arm model...")
    kinematics = _Kinematics()

    print(f"connecting to the arm at {args.ip}...")
    with _open_arm(args) as arm:
        error = check_kinematics(kinematics, arm, tolerance=args.fk_tolerance)
        offset = tool_offset(arm)
        if np.any(offset):
            # Worth saying out loud: it is accounted for in the check and plays
            # no part in the calibration, which is solved on flange poses.
            print(f"  controller has a TCP offset of {np.round(offset * 1000, 1)} mm, "
                  "which is accounted for")
        print(f"  forward kinematics agrees with the controller to {error * 1000:.1f} mm")

        print(f"opening the camera ({args.resolution})...")
        with _open_camera(args) as camera:
            serial = camera.serial
            if serial is None:
                raise HandEyeError(
                    "the camera's serial number could not be read, and it is "
                    "what the calibration is filed under. Pass --serial — it is "
                    "printed on the camera's back panel."
                )
            lens = "fisheye" if camera.calibration.fisheye else "pinhole"
            print(f"  SN{serial} at {camera.size[0]}x{camera.size[1]}, {lens} lens")
            intrinsics = Intrinsics.from_stereo(camera.calibration)

            # DEBUG (temporary)
            global _DEBUG
            _DEBUG = _DebugLog()
            _DEBUG.intrinsics = intrinsics
            try:
                _DEBUG.meta(args, spec, arm, camera, intrinsics)
            except Exception as error:  # noqa: BLE001
                print(f"  DEBUG: could not record meta: {error!r}")

            preview = Preview(
                camera.size, enabled=not args.no_window, scale=args.display_scale
            )
            try:
                capture = _capture_manual if args.manual else _capture_automatic
                observations = capture(
                    arm, camera, detector, kinematics, args, preview
                )
            finally:
                # Closed before the report, so the solver's numbers are not
                # printed behind a window the TA now has to go and dismiss.
                preview.close()

    print(f"\n{len(observations)} usable poses")
    if len(observations) < 3:
        raise HandEyeError(
            f"only {len(observations)} poses produced a usable view of the "
            "board, and the solver needs at least 3 (and really wants 15). "
            "Check that the board is in the camera's view from where the arm "
            "started, and that it is lit well enough to detect."
        )

    print("solving...")
    board_poses = [
        fit_board_pose(observation, intrinsics, detector.object_points)
        for observation in observations
    ]
    T_base_cam, T_flange_board = solve(
        [observation.flange for observation in observations], board_poses
    )
    # DEBUG (temporary)
    try:
        _DEBUG.solve(observations, board_poses, T_base_cam, T_flange_board,
                     intrinsics, detector.object_points)
    except Exception as error:  # noqa: BLE001
        print(f"  DEBUG: could not record the solve: {error!r}")
    try:
        T_base_cam, T_flange_board, stats = refine(
            observations,
            intrinsics,
            detector.object_points,
            T_base_cam,
            T_flange_board,
        )
    except Exception:
        _DEBUG.failure()  # DEBUG (temporary)
        raise

    spread = None
    if len(observations) >= 4 and not args.no_leave_one_out:
        print("checking how much one pose matters...")
        spread, _ = leave_one_out(
            observations,
            intrinsics,
            detector.object_points,
            T_base_cam,
            T_flange_board,
        )
        stats["leave_one_out_spread_m"] = spread

    _report(T_base_cam, T_flange_board, stats, spread)

    handeye = HandEye(
        T_base_cam=T_base_cam,
        T_flange_board=T_flange_board,
        intrinsics=intrinsics,
        board=spec,
        serial=serial,
        resolution=args.resolution,
        station=args.station,
        stats=stats,
    )

    if args.dry_run:
        print("  --dry-run: nothing written.")
        return 0

    handeye.tag = _reference_after_run(args)
    return _file(handeye, args)


def _reference_after_run(args):
    """Record the table tag for a calibration just measured, if it can be seen.

    Done here because this is the moment the calibration is known good. Nothing
    that goes wrong with it may cost the calibration just measured.
    """
    print(f"\nlooking for table tag {TAG_ID}...")
    try:
        with _open_tag_camera(args) as camera:
            reference = record_reference(camera, length=args.tag_length / 1000.0)
    except Exception as error:
        print(f"  ! could not look for it: {error}")
        reference = None
    if reference is None:
        print(
            "  ! no table tag recorded, so a knocked camera on this bench will not "
            "be noticed\n    or recovered from. If there is a tag and the arm is "
            "in front of it, move the arm\n    and run `register-tag`. If there "
            "is none, `python -m xarm7_lib.fiducials tag`\n    prints one."
        )
    else:
        _report_reference(reference)
    return reference


def _open_tag_camera(args):
    from .camera import ZedOpenCapture

    return ZedOpenCapture(
        resolution=TAG_RESOLUTION, serial=args.serial, controls=CAMERA_CONTROLS
    )


def _report_reference(reference):
    """Say what was recorded, and whether it looks like the tag that was printed."""
    corners = reference.points
    edge = float(
        np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))
    )
    centre = corners.mean(axis=0)
    # The shortest side, because lying flat foreshortens one way, and that way
    # decides how well the corners can be found.
    side_px = float(np.min(np.linalg.norm(
        np.asarray(reference.left) - np.roll(reference.left, 1, axis=0), axis=1
    )))
    print(
        f"  table tag {reference.tag_id} recorded at {reference.resolution}: "
        f"{np.linalg.norm(centre):.2f} m away, {side_px:.0f} px on its shortest side"
    )
    if abs(edge - reference.length) > 0.1 * reference.length:
        print(
            f"  ! it measures {edge * 1000:.1f} mm across, not the "
            f"{reference.length * 1000:.1f} mm declared. Recovery does not depend "
            "on the size, but\n    check this is the tag you meant."
        )
    if side_px < MIN_TAG_SIDE_PX:
        print(
            "  ! that is small enough to be unreliable. Move the tag closer to "
            "the camera, prop\n    it up to face the camera rather than lying "
            "flat, or print a bigger one with\n    `python -m xarm7_lib.fiducials tag --length 70`."
        )


def _file(handeye, args):
    """Write a calibration where it belongs, and say what to do next."""
    serial = handeye.serial
    destination = _destination(args, serial)
    try:
        written = handeye.save(destination)
    except HandEyeError as error:
        if args.system:
            raise HandEyeError(
                f"{error}\n"
                "That is the shared location inside the installation, so writing "
                "there needs the\naccess a TA has and a student does not. Drop "
                "--system to file it under\n"
                f"{USER_DIR} for this machine's accounts -- which\nis the default, "
                "and prints the `sudo` line that moves it into the\ninstallation "
                "afterwards -- or use --output to put it somewhere of your "
                "choosing."
            ) from error
        raise HandEyeError(
            f"{error}\n"
            "That is this account's own configuration directory, so something "
            "other than\naccess is wrong with it. --output puts the result "
            "somewhere of your choosing."
        ) from error

    print(f"  filed at {written}")
    if args.system:
        print(
            "\nEvery student on this station now gets it from `load_handeye()`."
        )
    else:
        print("\nThis account now gets it from `load_handeye()`.")
    if not args.system and not args.output:
        # The one thing left that this process cannot do itself, since it is not
        # running as the account that owns the installation. Printed last so it
        # is the line still on screen when the TA looks back at the terminal.
        print(
            "\nTo hand it to every student on this station, move it into the "
            "installation:\n"
            f"    {_promote_command(written, serial)}\n"
            "Until then it is this account's calibration and nobody else's. It "
            "also outranks\nthe shared one for this account, so delete it if "
            "you later want to check what\nthe class is actually reading."
        )
    return 0


# ----------------------------------------------------------------------
# register-tag
# ----------------------------------------------------------------------


def cmd_register_tag(args):
    """Add the table tag to a calibration already filed."""
    with _open_tag_camera(args) as camera:
        serial = camera.serial or args.serial
        if serial is None:
            raise HandEyeError(
                "the camera's serial number could not be read, and it is what "
                "the calibration is filed under. Pass --serial."
            )
        handeye = find_filed(serial)
        print(f"{handeye!r}")
        print(f"  from {handeye.source}, measured {handeye.created}")
        print(f"recording tag {args.id} at {TAG_RESOLUTION}...")
        reference = record_reference(
            camera, args.id, args.dict, args.length / 1000.0
        )
    if reference is None:
        raise HandEyeError(
            f"tag {args.id} ({args.dict}) was not seen steadily in both eyes. "
            "Check it is in view, in the light, and clear of the arm."
        )
    _report_reference(reference)

    # Recording a new reference is how "the camera has moved" gets forgotten, so
    # it should not happen by accident on a camera that already has.
    if handeye.tag is not None and handeye.tag.tag_id == reference.tag_id and (
        handeye.tag.resolution == reference.resolution
    ):
        shift = fit_camera_shift(handeye.tag, reference.left, reference.right)
        if shift.moved and not args.force:
            raise HandEyeError(
                f"the camera has moved {shift.describe()} since the reference "
                f"already filed here ({handeye.tag.created}), so this calibration "
                "is no longer good and recording a new reference would hide that. "
                "Redo the calibration with `run`, which records the tag as well. "
                "If it was the tag that was moved and the camera is known to be "
                "where it was, --force does it anyway."
            )

    print(
        "\nThis vouches for the calibration as it stands: from here on, the "
        "camera being where\nit is now is what counts as it not having moved. "
        "If it may have been knocked since it\nwas calibrated, recalibrate with "
        "`run` instead."
    )
    handeye.tag = reference
    return _file(handeye, args)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------


def _add_hardware_arguments(parser):
    group = parser.add_argument_group("hardware")
    group.add_argument(
        "--ip",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="controller address, e.g. 192.168.1.185. Defaults to $ROBOT_IP.",
    )
    group.add_argument(
        "--resolution", default="2K", help="2K, FHD, HD or VGA (default HD)"
    )
    group.add_argument("--serial", help="which camera, if several are attached")
    group.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"frames averaged per pose (default {DEFAULT_FRAMES})",
    )
    group.add_argument(
        "--no-window", action="store_true", help="no preview window, terminal only"
    )
    group.add_argument(
        "--display-scale",
        type=float,
        default=PREVIEW_SCALE,
        help=f"shrink the preview window (default {PREVIEW_SCALE})",
    )


def _add_destination_arguments(parser):
    where = parser.add_argument_group("where the result goes")
    where.add_argument("-o", "--output", help="write here instead")
    where.add_argument(
        "--user",
        action="store_true",
        help=f"write under {USER_DIR} — the default, and already implied",
    )
    where.add_argument(
        "--system",
        action="store_true",
        help=(
            f"write straight into the installation ({INSTALLED_DIR}) instead, "
            "which needs write access to it"
        ),
    )
    return where


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m xarm7_lib.calibrate_handeye",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="calibrate: move the arm, watch the board, solve, file the result",
        description=(
            "Put the arm where the camera can see the board on the gripper, "
            "then run this. The sweep starts from wherever the arm already is."
        ),
    )
    add_board_arguments(run)
    _add_hardware_arguments(run)
    sweep = run.add_argument_group("the sweep")
    sweep.add_argument(
        "--poses",
        type=int,
        default=DEFAULT_POSES,
        help=f"how many poses to visit (default {DEFAULT_POSES})",
    )
    sweep.add_argument(
        "--manual",
        action="store_true",
        help="push the arm to each pose by hand instead of generating a sweep",
    )
    sweep.add_argument(
        "--speed", type=float, default=DEFAULT_SPEED, help="rad/s between poses"
    )
    sweep.add_argument(
        "--settle",
        type=float,
        default=DEFAULT_SETTLE,
        help=f"seconds to stop ringing after a move (default {DEFAULT_SETTLE})",
    )
    sweep.add_argument("--seed", type=int, help="make the generated sweep repeatable")
    sweep.add_argument(
        "--no-guard",
        action="store_true",
        help="don't check poses for self-collision or against the safety box",
    )
    sweep.add_argument(
        "--fk-tolerance",
        type=float,
        default=FK_TOLERANCE,
        metavar="M",
        help=argparse.SUPPRESS,
    )
    sweep.add_argument(
        "--no-leave-one-out",
        action="store_true",
        help="skip the refit-per-dropped-pose check, which is the slow part",
    )
    run.add_argument(
        "--tag-length",
        type=float,
        default=TAG_LENGTH * 1000,
        metavar="MM",
        help=(
            "side of the table tag recorded at the end, if it is not the default "
            f"{TAG_LENGTH * 1000:g}. Only a check that it is the tag you meant"
        ),
    )
    where = _add_destination_arguments(run)
    where.add_argument(
        "--dry-run",
        action="store_true",
        help="solve and report, write nothing — the right first run on a bench",
    )
    where.add_argument("--station", help="a label for this bench, stored with the result")
    run.set_defaults(handler=cmd_run)

    register = commands.add_parser(
        "register-tag",
        help="record the table tag for a calibration already filed",
        description=(
            "`run` does this itself. This is for a calibration filed before the "
            "tag existed, or one whose tag has been replaced."
        ),
    )
    add_tag_identity_arguments(register)
    register.add_argument("--serial", help="which camera, if several are attached")
    register.add_argument(
        "--force",
        action="store_true",
        help="record it even though the existing reference says the camera has moved",
    )
    _add_destination_arguments(register)
    register.set_defaults(handler=cmd_register_tag)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    try:
        return args.handler(args)
    except HandEyeError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure where the ZED is, in the robot's coordinates. A TA runs this once per bench.

    python -m xarm7_lib.calibrate_handeye board          # the board to print and stick on
    python -m xarm7_lib.calibrate_handeye run --dry-run  # solve, report, write nothing
    python -m xarm7_lib.calibrate_handeye run            # ... and file the result
    python -m xarm7_lib.calibrate_handeye verify         # check a filed one, live
    python -m xarm7_lib.calibrate_handeye self-test      # the solver, no hardware

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
by that same few percent with residuals that look perfect -- the solver has no
way to know the board is not the size it was told. So `board` writes a PDF
whose page is a standard paper size with the board on it at exact scale,
prints the spec *onto the board itself*, and tells you to caliper a square
before trusting it.

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
import struct
import sys
import time
import zlib

import cv2
import numpy as np

from .handeye import (
    INSTALLED_DIR,
    USER_DIR,
    BoardSpec,
    HandEye,
    HandEyeError,
    Intrinsics,
    Observation,
    fit_board_pose,
    filename,
    invert,
    leave_one_out,
    load_handeye,
    pose,
    refine,
    reprojection_errors,
    rotation_angle,
    solve,
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
DEFAULT_SETTLE = 0.6  # s after a move, for the arm to stop ringing
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

# The marker is printed inside the square, so its size is a fraction of the
# square rather than a length of its own: the white quiet zone the detector
# needs around a marker has to stay proportional when the board is resized.
# Three quarters is what OpenCV's own samples use.
MARKER_RATIO = 0.75
DEFAULT_SQUARE_LENGTH = 0.020  # m

DEFAULT_BOARD = BoardSpec(
    squares_x=5,
    squares_y=7,
    square_length=DEFAULT_SQUARE_LENGTH,
    marker_length=DEFAULT_SQUARE_LENGTH * MARKER_RATIO,
)
DEFAULT_DPI = 600
DEFAULT_MARGIN = 10.0  # mm of white around the printed board


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
# The board
# ----------------------------------------------------------------------


def aruco_dictionary(name):
    """One of OpenCV's predefined dictionaries, by name."""
    code = getattr(cv2.aruco, name, None)
    if not isinstance(code, int):
        available = sorted(
            attribute
            for attribute in dir(cv2.aruco)
            if attribute.startswith("DICT_")
        )
        raise HandEyeError(
            f"unknown ArUco dictionary {name!r}. Available: {', '.join(available)}"
        )
    return cv2.aruco.getPredefinedDictionary(code)


def charuco_board(spec):
    """Build the OpenCV board for a `BoardSpec`, with the geometry cross-checked.

    `BoardSpec.object_points` computes the corner layout in plain numpy so that
    reading a calibration back needs no OpenCV. That makes it a second
    implementation of something OpenCV already knows, so it is checked against
    OpenCV's here -- the one place both are available. A mismatch would be a
    silent corner-ordering bug, which is the kind that produces a confident
    wrong answer.
    """
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length,
        spec.marker_length,
        aruco_dictionary(spec.dictionary),
    )
    if spec.legacy:
        # Boards generated by OpenCV before 4.6 have the marker grid shifted by
        # one square. Only for a board printed from an older install.
        board.setLegacyPattern(True)

    theirs = np.asarray(board.getChessboardCorners(), dtype=float)
    ours = spec.object_points()
    if theirs.shape != ours.shape or not np.allclose(theirs, ours, atol=1e-6):
        raise HandEyeError(
            "BoardSpec.object_points disagrees with OpenCV's "
            "getChessboardCorners for this board, so the corner ids would not "
            "line up with the geometry. This is a library bug, not a setup "
            "problem."
        )
    return board


def board_image(spec, dpi=DEFAULT_DPI, margin=DEFAULT_MARGIN, caption=True):
    """Render the board for printing. Reports `(image, dpi)`.

    The squares come out `spec.square_length` across when the result is printed
    at the dpi reported back, which is the only reason to care about dpi here.
    That is not quite the `dpi` asked for: a square has to be a whole number of
    pixels, so the resolution is nudged to the nearest one that makes it so and
    the exact physical size is preserved instead. Printing at the reported dpi
    is what `write_board_png` and `write_board_pdf` arrange, so nothing has to
    be done about it.

    Rounding the square rather than the whole board is what keeps every square
    identical -- and OpenCV lays the marker bits out inside each square with
    integer pixel arithmetic, so a square that is a fractional number of pixels
    across can overflow its cell and abort the render outright.

    With `caption`, the spec is drawn in the bottom margin, so the physical
    board says what it is. That line is what lets a TA holding the board tell
    whether it is the board `run` is being told about -- and the sizes are
    there to be checked against a caliper, because a board scaled by the print
    dialog is the one error that leaves no trace in the residuals.
    """
    square_px = max(1, int(round(spec.square_length * dpi / 0.0254)))
    dpi = square_px / spec.square_length * 0.0254
    pixels_per_metre = dpi / 0.0254

    width = spec.squares_x * square_px
    height = spec.squares_y * square_px
    # marginSize=0 because the margin is added below, where it is in
    # millimetres rather than whatever fraction of the image OpenCV picks.
    squares = charuco_board(spec).generateImage(
        (width, height), marginSize=0, borderBits=1
    )

    pad = int(round(margin / 1000.0 * pixels_per_metre))
    band = 0
    text = spec.describe()
    scale = thickness = baseline = 0
    if caption:
        # Aim for roughly 4 mm of cap height, then shrink if the line would run
        # off the sheet. The caption is allowed the full width including the
        # margins, because on a small board it would otherwise shrink to
        # something a caliper-wielding TA cannot read.
        room = width + 2 * pad
        scale = max(0.3, 4.0 / 1000.0 * pixels_per_metre / 22.0)
        thickness = max(1, int(round(scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        if text_width > room:
            scale *= room / text_width
            thickness = max(1, int(round(scale * 2)))
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
        band = text_height + baseline + pad

    canvas = np.full((height + 2 * pad + band, width + 2 * pad), 255, dtype=np.uint8)
    canvas[pad:pad + height, pad:pad + width] = squares
    if caption:
        cv2.putText(
            canvas,
            text,
            (max(0, (canvas.shape[1] - text_width) // 2), height + 2 * pad + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            0,
            thickness,
            cv2.LINE_AA,
        )
    return canvas, dpi


PAPER_SIZES = (
    # name, width, height, in mm. Tried in this order; the first the board fits
    # on is the page the PDF gets, so the page matches the paper actually in the
    # tray and nothing in the print dialog is tempted to rescale it.
    ("Letter", 215.9, 279.4),
    ("A4", 210.0, 297.0),
    ("Legal", 215.9, 355.6),
    ("Tabloid", 279.4, 431.8),
    ("A3", 297.0, 420.0),
)


def write_board_png(path, image, dpi):
    """Write the board as a PNG that says how big it is.

    OpenCV writes no `pHYs` chunk, and a PNG without one has no physical size:
    every viewer falls back to 72 dpi, so a 600 dpi board claims to be eight
    times its real width and "100%" prints it eight times too big. The chunk is
    nine bytes and goes straight after IHDR, so it is added here rather than
    taking on an imaging dependency for it.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HandEyeError(f"could not encode {path}")
    data = buffer.tobytes()

    # 8-byte signature, then IHDR: 4 length + 4 type + 13 data + 4 crc.
    end_of_ihdr = 8 + 25
    if data[8 + 4:8 + 8] != b"IHDR":
        raise HandEyeError("encoded PNG does not start with IHDR, which is a bug")
    pixels_per_metre = int(round(dpi / 0.0254))
    body = b"pHYs" + struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1)
    chunk = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(data[:end_of_ihdr] + chunk + data[end_of_ihdr:])


def write_board_pdf(path, image, dpi):
    """Write the board as a PDF page at exact scale, and report the page size.

    This is the format to print. A PDF carries the board's size in points
    rather than in metadata a viewer may ignore, and the page is a standard
    paper size with the board centred on it, so "actual size" and "100%" and
    "no scaling" all mean the same thing and "fit to page" has nothing to
    shrink. Returns the paper name, or None if the board needed a page of its
    own because it fits no standard size.

    Written by hand because the alternative is a PDF library in the
    dependencies of a robot arm package. The whole file is one grayscale image
    XObject on one page, which is about as small as a PDF gets.
    """
    height, width = image.shape[:2]
    board_mm = (width / dpi * 25.4, height / dpi * 25.4)

    paper = next(
        (name for name, w, h in PAPER_SIZES if board_mm[0] <= w and board_mm[1] <= h),
        None,
    )
    if paper is None:
        page_mm = board_mm
    else:
        page_mm = next((w, h) for name, w, h in PAPER_SIZES if name == paper)

    # PDF user space is 1/72 inch, which is what "72 dpi" has always meant.
    to_points = 72.0 / 25.4
    page = (page_mm[0] * to_points, page_mm[1] * to_points)
    drawn = (board_mm[0] * to_points, board_mm[1] * to_points)
    origin = ((page[0] - drawn[0]) / 2.0, (page[1] - drawn[1]) / 2.0)

    samples = zlib.compress(np.ascontiguousarray(image).tobytes(), 9)
    content = (
        f"q {drawn[0]:.4f} 0 0 {drawn[1]:.4f} {origin[0]:.4f} {origin[1]:.4f} cm "
        "/Im0 Do Q\n"
    ).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page[0]:.4f} {page[1]:.4f}] "
            "/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(samples)} >>\nstream\n"
        ).encode("ascii") + samples + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    start_xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % start_xref

    path.write_bytes(bytes(out))
    return paper


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
    part, and it is gone by the time the arm has moved on.
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

    def show(self, image, corners=None, extra=None, predicted=None, note=()):
        """Draw one frame. Returns the key pressed in the window, or -1.

        `corners` are the detections the solver will use, `extra` the ones it
        cannot (seen in one eye only), and `predicted` where the calibration
        says the corners should be -- which is the whole of what `verify` has
        to say, drawn on the thing it is talking about.

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
        for points, colour, radius in (
            (extra, (0, 180, 255), 3),
            (predicted, (255, 0, 255), 6),
            (corners, (0, 255, 0), 3),
        ):
            if points is None:
                continue
            # Open circles for the prediction, filled for a measurement: where
            # the two disagree the pair reads as a target and a miss.
            thickness = 1 if points is predicted else -1
            for u, v in np.asarray(points, dtype=float).reshape(-1, 2) * self.scale:
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                cv2.circle(canvas, (int(round(u)), int(round(v))), radius,
                           colour, thickness, cv2.LINE_AA)
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
        the thing that stops the window going grey. Returns the key pressed.
        """
        self.key = -1
        if not self.enabled:
            time.sleep(seconds)
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
# board
# ----------------------------------------------------------------------


def cmd_board(args):
    spec = _spec_from_args(args)
    # Not args.dpi: board_image nudges it to whatever keeps a square a whole
    # number of pixels, and it is the nudged one that makes the squares come
    # out the size the spec says.
    image, dpi = board_image(
        spec, dpi=args.dpi, margin=args.margin, caption=not args.no_caption
    )

    output = pathlib.Path(args.output)
    paper = None
    if output.suffix.lower() == ".pdf":
        paper = write_board_pdf(output, image, dpi)
    else:
        write_board_png(output, image, dpi)

    width = spec.squares_x * spec.square_length * 1000
    height = spec.squares_y * spec.square_length * 1000
    sheet = (image.shape[1] / dpi * 25.4, image.shape[0] / dpi * 25.4)
    print(f"wrote {output}  ({image.shape[1]}x{image.shape[0]} px at {dpi:.0f} dpi)")
    print(f"  {spec.describe()}")
    print(f"  board area {width:.1f} x {height:.1f} mm, {spec.n_corners} corners")
    print(f"  printed size {sheet[0]:.1f} x {sheet[1]:.1f} mm including the margin")
    if paper is not None:
        print(f"  on one {paper} page, centred")
    elif output.suffix.lower() == ".pdf":
        print(
            "  on a page of its own: this board fits no standard paper size, so "
            "the\n  print dialog will offer to scale it. Don't let it — print it "
            "on larger\n  paper, or use a smaller --square-length."
        )

    print(
        "\nPrint it at 100% — 'actual size', not 'fit to page' and not 'shrink to\n"
        "printable area'. Then measure one square with a caliper and check it "
        f"against\n{spec.square_length * 1000:.1f} mm.\n"
        "A board scaled by the print dialog gives a calibration that is wrong "
        "by that\nsame scale, with residuals that look perfect. The sizes "
        "printed on the board\nare there so you can check them.\n"
    )
    if paper is None and output.suffix.lower() != ".pdf":
        print(
            "This is a PNG, which print dialogs scale to the paper by default. "
            "It now\ncarries its real size, so 100% is the right size — but "
            f"`-o {output.stem}.pdf` gives\nyou a page that is already the "
            "right size, which is harder to get wrong.\n"
        )
    print(
        "Mount it flat on the gripper — a board that flexes is a board whose "
        "corners are\nin the wrong place. It may not move relative to the "
        "flange after this point."
    )
    return 0


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

    return ZedOpenCapture(resolution=args.resolution, serial=args.serial)


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


def cmd_run(args):
    _require_ip(args)
    spec = _spec_from_args(args)
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
    T_base_cam, T_flange_board, stats = refine(
        observations,
        intrinsics,
        detector.object_points,
        T_base_cam,
        T_flange_board,
    )

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
    print(
        "Check it before you walk away:\n"
        "    python -m xarm7_lib.calibrate_handeye verify"
    )
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
# verify
# ----------------------------------------------------------------------


def cmd_verify(args):
    _require_ip(args)
    handeye = (
        HandEye.load(args.file) if args.file else load_handeye(serial=args.serial)
    )
    print(f"{handeye!r}")
    print(f"  from     {handeye.source}")
    print(f"  measured {handeye.created} against {handeye.board.describe()}")
    if handeye.stats:
        print(f"  fitted   {handeye.stats.get('poses')} poses, "
              f"{handeye.stats.get('rms_px', float('nan')):.3f} px RMS")

    detector = BoardDetector(handeye.board)
    kinematics = _Kinematics()

    print("\nconnecting...")
    # guard=False: this only ever reads the arm's position, so there is nothing
    # to check, and skipping the collision meshes makes it connect in a second.
    with _open_arm(args, guard=False) as arm:
        with _open_camera(args) as camera:
            if handeye.serial and camera.serial and camera.serial != handeye.serial:
                raise HandEyeError(
                    f"this calibration belongs to SN{handeye.serial} but the "
                    f"attached camera is SN{camera.serial}. They are not "
                    "interchangeable — the calibration is of one camera on one "
                    "bench."
                )
            if handeye.resolution and handeye.resolution != args.resolution:
                raise HandEyeError(
                    f"the calibration was made at {handeye.resolution} and this "
                    f"is {args.resolution}. The intrinsics differ between "
                    f"modes, so pass --resolution {handeye.resolution}."
                )

            print(
                "\nMove the arm by hand or from another terminal and watch the "
                "error. These are\nposes the fit never saw, which is the only "
                "check that catches a calibration that\nis self-consistent and "
                "wrong. Ctrl-C to stop.\n"
            )
            print(
                f"  {'position':>12}  {'rotation':>9}  {'corners':>7}  "
                f"{'reprojection':>12}"
            )

            worst = 0.0
            preview = Preview(
                camera.size, enabled=not args.no_window, scale=args.display_scale
            )
            try:
                while True:
                    measured = arm.joint_values
                    flange = kinematics.flange_pose(measured)
                    observation = observe(camera, detector, flange, args.frames)
                    if observation is None:
                        _stream(camera, preview, 0.5, note=["board not in view"])
                        continue

                    seen = fit_board_pose(
                        observation, handeye.intrinsics, detector.object_points
                    )
                    predicted = handeye.board_pose(flange)
                    position = np.linalg.norm(seen[:3, 3] - predicted[:3, 3])
                    rotation = np.degrees(
                        rotation_angle(seen[:3, :3], predicted[:3, :3])
                    )
                    errors = reprojection_errors(
                        handeye.T_base_cam,
                        handeye.T_flange_board,
                        [observation],
                        handeye.intrinsics,
                        detector.object_points,
                    )
                    worst = max(worst, position)
                    rms = float(np.sqrt(np.mean(errors ** 2)))
                    print(
                        f"  {position * 1000:9.1f} mm  {rotation:7.2f}°  "
                        f"{len(observation):7d}  {rms:9.2f} px"
                    )
                    # Magenta is where the calibration says the corners are,
                    # green where they actually are. Overlapping rings mean it
                    # is right, and no table of residuals says it as plainly.
                    live, _ = camera.read_rectified()
                    in_camera = (
                        detector.object_points @ predicted[:3, :3].T + predicted[:3, 3]
                    )
                    preview.show(
                        live,
                        corners=observation.left,
                        predicted=handeye.intrinsics.project(in_camera, "left"),
                        note=[
                            f"{position * 1000:.1f} mm and {rotation:.2f}° off "
                            "the prediction",
                            f"{len(observation)} corners, {rms:.2f} px  "
                            "(magenta: predicted, green: measured)",
                        ],
                    )
                    if preview.hold(args.period) in (ord("q"), 27) or preview.closed:
                        break
            except KeyboardInterrupt:
                print()
            finally:
                preview.close()
                print(f"\n  worst position error seen: {worst * 1000:.1f} mm")
    return 0


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------


def _look_at(position, target):
    """A camera pose that puts `target` in the middle of the view.

    The camera frame is x right, y down, z forward, so this builds the rotation
    from the viewing direction and whichever way is down in the base frame.
    """
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    down = np.array([0.0, 0.0, -1.0])
    down = down - np.dot(down, forward) * forward
    down /= np.linalg.norm(down)
    return pose(np.stack([np.cross(down, forward), down, forward], axis=1), position)


def cmd_self_test(args):
    """Run the whole solver against a rig whose answer is known.

    No camera and no arm: real forward kinematics, real board geometry, a
    camera placed by hand, and corner noise. The one check that can be run
    anywhere, and it covers the part most likely to be silently wrong -- the
    conventions. Every transform here is composed in the same direction the
    real routine composes it, so a flipped inverse fails this.

    The thresholds are a regression check on the method, not a target. Over
    twelve seeds at the defaults the camera comes back within 0.01-0.07 degrees
    and 0.21-0.96 mm of truth, and the board within 0.08-0.24 degrees and
    0.06-0.33 mm, so they are set at about twice the worst of that: the draw
    alone should never fail this, while a mistake in the geometry misses by
    degrees and centimetres rather than by tenths.

    How much headroom that needs depends on the board. Orientation is recovered
    from how the corners spread across the image, so a board a third the size
    gives up roughly three times the angular precision -- the 30 mm board these
    thresholds were first set against held the board's rotation under 0.07
    degrees, where the 10 mm default reaches 0.24. Shrinking the default board
    again means measuring the range again rather than assuming these still fit.
    """
    rng = np.random.default_rng(args.seed)
    spec = _spec_from_args(args)
    detector = BoardDetector(spec)
    # Mounted by its middle, as a board on a gripper would be, so the test does
    # not accidentally depend on the board frame sitting at a corner.
    object_points = detector.object_points - detector.object_points.mean(axis=0)

    kinematics = _Kinematics()
    intrinsics = Intrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0, baseline=0.12)

    seed = np.array([0.0, -0.35, 0.0, 0.75, 0.0, 0.55, 0.0])
    T_base_cam = _look_at([1.0, -0.7, 0.8], kinematics.flange_pose(seed)[:3, 3])
    T_flange_board = pose([0.02, 0.03, 0.4], [0.0, 0.0, 0.085])

    print(f"board: {spec.describe()}  ({spec.n_corners} corners)")
    print(f"noise: {args.noise} px per corner per eye")
    print(f"camera truth: {np.round(T_base_cam[:3, 3], 4)} m\n")

    poses = generate_poses(seed, args.poses, rng=rng)
    observations = []
    for q in poses:
        flange = kinematics.flange_pose(q)
        T_cam_board = invert(T_base_cam) @ flange @ T_flange_board
        in_camera = object_points @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
        if np.any(in_camera[:, 2] < 0.2):
            continue  # behind or on top of the camera; not a pose a bench has
        observations.append(
            Observation(
                flange=flange,
                ids=np.arange(len(object_points)),
                left=intrinsics.project(in_camera, "left")
                + rng.normal(0, args.noise, (len(object_points), 2)),
                right=intrinsics.project(in_camera, "right")
                + rng.normal(0, args.noise, (len(object_points), 2)),
            )
        )
    print(f"{len(observations)} synthetic poses")

    board_poses = [
        fit_board_pose(observation, intrinsics, object_points)
        for observation in observations
    ]
    guess_cam, guess_board = solve(
        [observation.flange for observation in observations], board_poses
    )
    fitted_cam, fitted_board, stats = refine(
        observations, intrinsics, object_points, guess_cam, guess_board
    )
    spread, _ = leave_one_out(
        observations, intrinsics, object_points, fitted_cam, fitted_board
    )

    def compare(name, fitted, truth):
        rotation = np.degrees(rotation_angle(fitted[:3, :3], truth[:3, :3]))
        translation = 1000 * np.linalg.norm(fitted[:3, 3] - truth[:3, 3])
        print(f"  {name:<24} {rotation:8.4f}°  {translation:8.3f} mm")
        return rotation, translation

    print(f"\n  {'':<24} {'rotation':>8}   {'position':>8}")
    compare("closed form, camera", guess_cam, T_base_cam)
    camera_rotation, camera_translation = compare("refined, camera", fitted_cam, T_base_cam)
    board_rotation, board_translation = compare("refined, board", fitted_board, T_flange_board)

    _report(fitted_cam, fitted_board, stats, spread)

    # The storage round trip, in the same breath: a calibration that solves and
    # then cannot be read back is no use.
    handeye = HandEye(
        fitted_cam, fitted_board, intrinsics, spec, serial="selftest", stats=stats
    )
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / filename("selftest")
    reloaded = HandEye.load(handeye.save(path))
    round_trip = np.abs(reloaded.T_base_cam - fitted_cam).max()
    probe = np.array([[0.1, 0.2, 1.0]])
    reversed_probe = np.abs(
        reloaded.points_to_camera(reloaded.points_to_base(probe)) - probe
    ).max()

    failures = []
    if camera_rotation > args.max_rotation:
        failures.append(
            f"camera rotation off by {camera_rotation:.4f}°, allowed {args.max_rotation}"
        )
    if camera_translation > args.max_translation:
        failures.append(
            f"camera position off by {camera_translation:.3f} mm, allowed "
            f"{args.max_translation}"
        )
    if board_rotation > args.max_rotation or board_translation > args.max_translation:
        failures.append(
            f"board pose off by {board_rotation:.4f}° / {board_translation:.3f} mm"
        )
    if round_trip > 1e-9:
        failures.append(f"storage round trip changed the transform by {round_trip}")
    if reversed_probe > 1e-9:
        failures.append(f"points_to_base/points_to_camera do not invert: {reversed_probe}")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASSED  camera recovered to {camera_rotation:.4f}° and "
        f"{camera_translation:.3f} mm; storage round trips exactly."
    )
    return 0


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------


def _squares(value):
    """`5x7`, as the board's square counts."""
    try:
        x, y = value.lower().split("x")
        return int(x), int(y)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected something like 5x7, got {value!r}"
        ) from None


def _spec_from_args(args):
    square_length = args.square_length / 1000.0
    # Scaled off the square unless asked for outright, so that changing
    # --square-length alone gives a board in proportion rather than one whose
    # markers overflow their squares.
    marker_length = (
        square_length * MARKER_RATIO
        if args.marker_length is None
        else args.marker_length / 1000.0
    )
    return BoardSpec(
        squares_x=args.squares[0],
        squares_y=args.squares[1],
        square_length=square_length,
        marker_length=marker_length,
        dictionary=args.dict,
        legacy=args.legacy,
    )


def _add_board_arguments(parser, default=DEFAULT_BOARD):
    group = parser.add_argument_group("the board on the gripper")
    group.add_argument(
        "--squares",
        type=_squares,
        default=(default.squares_x, default.squares_y),
        metavar="NxM",
        help=f"squares across and down (default {default.squares_x}x{default.squares_y})",
    )
    group.add_argument(
        "--square-length",
        type=float,
        default=default.square_length * 1000,
        metavar="MM",
        help=f"square side in mm (default {default.square_length * 1000:g})",
    )
    group.add_argument(
        "--marker-length",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "marker side in mm, smaller than the square. Defaults to "
            f"{MARKER_RATIO:g} of --square-length, which is "
            f"{default.marker_length * 1000:g} mm for the default board"
        ),
    )
    group.add_argument(
        "--dict",
        default=default.dictionary,
        help=f"ArUco dictionary (default {default.dictionary})",
    )
    group.add_argument(
        "--legacy",
        action="store_true",
        help="the board was generated by OpenCV before 4.6, whose marker grid is offset",
    )


def _add_hardware_arguments(parser):
    group = parser.add_argument_group("hardware")
    group.add_argument(
        "--ip",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="controller address, e.g. 192.168.1.185. Defaults to $ROBOT_IP.",
    )
    group.add_argument(
        "--resolution", default="HD", help="2K, FHD, HD or VGA (default HD)"
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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m xarm7_lib.calibrate_handeye",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser(
        "board",
        help="write the board to print and stick on the gripper",
        description=(
            "Render the ChArUco board at exact scale, with its own spec printed "
            "on it. Print at 100% and caliper a square before trusting it."
        ),
    )
    _add_board_arguments(board)
    board.add_argument(
        "-o",
        "--output",
        default="charuco_board.pdf",
        help=(
            "where to write it. A .pdf is a page at exact scale, which is what "
            "to print; any other extension writes an image instead"
        ),
    )
    board.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help=f"print resolution (default {DEFAULT_DPI})"
    )
    board.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        metavar="MM",
        help=f"white border in mm (default {DEFAULT_MARGIN:.0f})",
    )
    board.add_argument(
        "--no-caption",
        action="store_true",
        help="leave the spec off the printout (don't: it is how you check the board)",
    )
    board.set_defaults(handler=cmd_board)

    run = commands.add_parser(
        "run",
        help="calibrate: move the arm, watch the board, solve, file the result",
        description=(
            "Put the arm where the camera can see the board on the gripper, "
            "then run this. The sweep starts from wherever the arm already is."
        ),
    )
    _add_board_arguments(run)
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
    where = run.add_argument_group("where the result goes")
    where.add_argument(
        "--dry-run",
        action="store_true",
        help="solve and report, write nothing — the right first run on a bench",
    )
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
    where.add_argument("--station", help="a label for this bench, stored with the result")
    run.set_defaults(handler=cmd_run)

    verify = commands.add_parser(
        "verify",
        help="check a filed calibration against what the camera sees now",
        description=(
            "Compare the board pose the camera measures against the one "
            "forward kinematics predicts, live, at whatever poses you like."
        ),
    )
    _add_hardware_arguments(verify)
    verify.add_argument("--file", help="check this file instead of the filed one")
    verify.add_argument(
        "--period", type=float, default=0.5, help="seconds between readings"
    )
    verify.set_defaults(handler=cmd_verify)

    self_test = commands.add_parser(
        "self-test",
        help="run the solver against a known answer; no camera, no arm",
        description=(
            "Synthetic rig with real kinematics and real board geometry. Checks "
            "the solver, the conventions and the storage round trip."
        ),
    )
    _add_board_arguments(self_test)
    self_test.add_argument("--poses", type=int, default=16)
    self_test.add_argument(
        "--noise", type=float, default=0.3, metavar="PX", help="corner noise, px"
    )
    self_test.add_argument("--seed", type=int, default=7)
    # Roughly twice the worst seen over ten seeds at the defaults (0.044 deg,
    # 0.75 mm) -- see SELF_TEST_TOLERANCE. Loose enough not to be a coin flip
    # on the draw, tight enough that a broken convention cannot pass.
    # Twice the worst seen over twelve seeds at the default board; see
    # cmd_self_test's docstring for why this tracks the board's size.
    self_test.add_argument("--max-rotation", type=float, default=0.50, metavar="DEG")
    self_test.add_argument("--max-translation", type=float, default=2.0, metavar="MM")
    self_test.set_defaults(handler=cmd_self_test)

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

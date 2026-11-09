"""Where the camera is, in the robot's coordinates.

The ZED on this bench is bolted down relative to the arm's base, so one rigid
transform carries everything the camera measures into robot coordinates. That
transform is what `StereoCalibration` stops short of: the factory calibration
describes the camera's own geometry -- two eyes, their intrinsics, the baseline
between them -- and triangulating a point with it gives metres in the *camera's*
frame. Reaching for that point needs the same metres in the *base* frame, and
nothing about the camera knows where the camera is.

So it is measured once per bench, by a TA, with `calibrate_handeye.py`, and
filed where every student on that station reads the same answer:

    from xarm7_lib import load_handeye
    handeye = load_handeye()
    base_points = handeye.points_to_base(camera_points)

`load_handeye` finds the file; students never name a path and never run the
calibration.

What this module holds is the arithmetic and the file format. The capture side
-- the board, the detector, the arm -- lives in `calibrate_handeye.py`, which
nothing here imports.


The geometry
------------

A ChArUco board is stuck to the gripper and seen from the fixed camera at a
number of arm poses. Two things are unknown and neither can be measured with a
ruler: where the camera is, and where the board sits on the flange. Both come
out of one relation, which holds at every pose `i`:

    T_base_cam . T_cam_board(i)  =  T_base_flange(i) . T_flange_board
         X      .      A(i)      =         B(i)      .       Y

`B(i)` is forward kinematics, so it is known exactly. `A(i)` is what the camera
sees. `X` and `Y` are the unknowns, and there are only two of them -- **12
parameters in total** -- because once they are fixed, every board pose follows
from `A(i) = X^-1 B(i) Y`. The per-pose board poses are not free parameters.

That last point is what shapes `refine`. A bundle adjustment would carry six
numbers per pose and fit them alongside the two transforms; here there is
nothing to carry, and the right final step is to drive all 12 parameters
against every detected corner in both eyes at once. The per-pose poses are
needed only to start `solve` off, and an error in one of them costs nothing
afterwards.

`solve` does that start with a closed form and `refine` does the fitting. The
split matters: the closed form cannot use the corners, only the poses distilled
out of them, so it is accurate to a few millimetres and no better. The
refinement is what reaches the tenths of a millimetre, and it needs somewhere
to start from that is not 180 degrees out.


Rectified, throughout
---------------------

Every pixel here is a *rectified* pixel. Rectification is already the camera's
normal output path (`ZedOpenCapture.read_rectified`), and in that frame the
distortion is gone and both eyes share one focal length and one principal
point, so projection is `K (R p + t)` with the right eye shifted along x by the
baseline. No distortion model, no per-eye intrinsics, no cv2.

That is also why `Intrinsics` is copied into the stored file rather than looked
up from the camera when the file is read. Deriving it needs
`StereoCalibration.rectify()`, which needs OpenCV, and a student who only wants
to turn a point into base coordinates should not need OpenCV installed to do
it. The five numbers are small, they belong to the calibration that was
actually performed, and storing them makes the file self-sufficient.


When the camera gets knocked
----------------------------

A bump invalidates `T_base_cam`, and re-measuring it needs a TA, a board and
the arm. So a small tag taped to the table is recorded alongside the
calibration (`TagReference`), and whatever streams the camera -- the capstone's
`LightsaberDetector` -- compares what it sees now against that record. If the tag has moved in the picture, the camera has
moved, and `T_base_cam @ D` carries on working in the meantime, where `D` is the
camera's motion since the record -- `fit_camera_shift`.

Two things shape that fit.

It is differential. It never asks where the tag *is*, only how the same four
corners, seen by the same camera, have moved. Nothing about the tag's printed
size or the detector's bias on small markers enters, because both are in the
reference too. The tag's pose in the base frame is never used.

It is told that bumps mostly turn a camera. One small tag measures bearing and
range well and its own tilt badly: 45 mm of tag at 1.25 m cannot distinguish a
camera that turned from one that slid sideways along a sphere centred on the
tag. Left alone, that weak direction is what the noise goes into, and it is
worth a centimetre or more half a metre from the tag. A weak prior that the
optical centre stayed put settles it (`SHIFT_SIGMA_T`), and it applies only
across the line of sight, because range is something the stereo really does
measure.

So what comes back is a stopgap, accurate near the tag and less so away from
it, and everything that uses one says so out loud until a TA re-runs the
calibration. It also cannot tell a camera that moved from a tag that did.
"""

import copy
import dataclasses
import json
import os
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# A file named outright, which wins over both of the directories below. For a
# TA trying a candidate calibration without disturbing the one the class is
# using.
HANDEYE_ENV = "XARM7_HANDEYE"

# Where `calibrate_handeye run` writes by default, because the account running
# it usually cannot write to the installation below. Same idea as the local
# conf file in `calibration.py`: a copy on this machine outranks the shared
# one, because the only reasons to have one are to override it or to be on the
# way to becoming it.
USER_DIR = (
    pathlib.Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    / "xarm7_lib"
    / "handeye"
)

# Inside the installed package -- on a lab machine, under the read-only
# /opt/16384 tree. This is the shared one: filed by a TA, found by every
# student on the station, and not writable by them.
INSTALLED_DIR = pathlib.Path(__file__).resolve().parent / "data" / "handeye"

# Huber's knee, in pixels. A corner this far off the fit is treated as an
# outlier rather than as a measurement to satisfy, which keeps one bad
# detection in one pose from tilting all 12 parameters. Loose enough that
# ordinary sub-pixel scatter is still fitted as least squares.
OUTLIER_KNEE = 2.0

_FORMAT = 1  # of the stored file, in case the fields ever move

# The measurement noise `fit_camera_shift` assumes at the least, after averaging.
# Scatter measured across the frames raises it. Disparity gets more than a pixel
# does because the two eyes drift against each other as the camera warms up,
# which shifts the disparity and leaves the bearing alone.
SHIFT_SIGMA_PX = 0.1
SHIFT_SIGMA_DISPARITY = 0.3
# How far the optical centre is taken to have slid sideways in a bump, in
# metres. A camera pivoting on a mount 30 cm away moves about 5 mm per degree.
SHIFT_SIGMA_T = 0.010

# Below these it has not moved: not enough to matter, and small enough that
# warm-up can do it. Getting this wrong does harm, because a correction applied
# to a camera that never moved replaces a good calibration with noise.
#
# Bearing, where the tag's centre is in the picture, decides it, because it
# averages four corners and has the whole focal length behind it. Total rotation
# is no use for this: roll is measured across the width of a small tag, so it is
# by far the noisiest part, and a still camera on poor images reads up to 0.9
# degrees. So rotation only counts when it is more than noise can explain, which
# catches a turn about the line to the tag, the one bump bearing cannot see.
# Range is loose for the reason given at SHIFT_SIGMA_DISPARITY.
MOVED_BEARING = 0.1  # degrees
MOVED_ROTATION = 2.0  # degrees
MOVED_RANGE = 0.010  # m

# There is no upper limit. However far the camera has moved, a correction is
# used and the warning says how far off to expect it to be; a correction is a
# stopgap until the bench is recalibrated, not a measurement.

# How far from the tag `CameraShift.error_away` is quoted: about the size of the
# area a student works in.
ERROR_RADIUS = 0.5  # m


class HandEyeError(RuntimeError):
    """The calibration could not be found, read, made sense of, or solved."""


# ----------------------------------------------------------------------
# Rigid transforms
# ----------------------------------------------------------------------
#
# A pose is a plain 4x4 homogeneous matrix. Nothing here wraps it, because
# every consumer -- numpy, the JSON file, a student printing it -- wants the
# matrix itself.


def _as_pose(T, what="pose"):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise HandEyeError(f"{what} must be a 4x4 matrix, got shape {T.shape}")
    return T


def invert(T):
    """The inverse of a rigid transform, by transpose rather than by solve.

    `np.linalg.inv` on a 4x4 would give nearly the same answer, but it has no
    idea the top-left block is a rotation and will happily return something
    that isn't a rigid transform when the input has drifted. This cannot.
    """
    T = _as_pose(T)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def pose(rotation, translation):
    """Assemble a 4x4 from a rotation matrix (or rotation vector) and a translation."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape == (3,):
        rotation = Rotation.from_rotvec(rotation).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def _rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


def _nearest_rotation(M):
    """The rotation closest to `M`, which is the SVD with the determinant forced positive.

    Used wherever rotations are summed or least-squared and the result has to
    be put back on SO(3). The determinant guard is not decoration: without it a
    noisy sum can project onto a reflection, which looks like a rotation until
    it mirrors the whole reconstruction.
    """
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def rotation_angle(R1, R2):
    """The angle between two rotations, in radians."""
    return float(np.linalg.norm(_rotvec(np.asarray(R1).T @ np.asarray(R2))))


# ----------------------------------------------------------------------
# The target and the camera
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board, as the physical thing stuck to the gripper.

    Lengths are metres. `squares_x` by `squares_y` counts *squares*, as
    OpenCV's `CharucoBoard` does, so the corners the detector reports are the
    `(squares_x - 1) * (squares_y - 1)` interior ones.

    This is stored with the calibration because a calibration solved against
    one board means nothing for another, and a board whose squares printed at
    97% of the requested size is the classic way to get a confident answer
    that is wrong by 3%.
    """

    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str = "DICT_4X4_50"
    legacy: bool = False

    def __post_init__(self):
        if self.squares_x < 2 or self.squares_y < 2:
            raise HandEyeError(
                f"a board needs at least 2x2 squares, got "
                f"{self.squares_x}x{self.squares_y}"
            )
        if not 0 < self.marker_length < self.square_length:
            raise HandEyeError(
                f"marker length ({self.marker_length}) must be positive and "
                f"smaller than the square ({self.square_length}) -- the marker "
                "is printed inside the square"
            )

    @property
    def n_corners(self):
        """How many interior corners the detector can report."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def object_points(self):
        """The interior corners in the board's own frame, `(n_corners, 3)` metres.

        Row `k` is the corner whose ChArUco id is `k`, which is what lets a
        detection be matched to its geometry by id alone.

        This reproduces `cv2.aruco.CharucoBoard.getChessboardCorners()` --
        origin at the board's corner, x fastest, z zero -- deliberately rather
        than calling it, so that reading a calibration back needs no OpenCV.
        `fiducials.charuco_board` checks the two against each other.
        """
        i, j = np.meshgrid(
            np.arange(1, self.squares_x), np.arange(1, self.squares_y), indexing="xy"
        )
        points = np.zeros((self.n_corners, 3))
        points[:, 0] = i.ravel() * self.square_length
        points[:, 1] = j.ravel() * self.square_length
        return points

    def describe(self):
        """One line naming the board, for a printout or a report."""
        return (
            f"{self.squares_x}x{self.squares_y}  "
            f"square {self.square_length * 1000:.1f} mm  "
            f"marker {self.marker_length * 1000:.1f} mm  "
            f"{self.dictionary}"
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    """The rectified stereo pair as a single pinhole plus a baseline.

    After rectification both eyes share `fx`, `fy`, `cx` and `cy`, and the
    right eye is the left one translated along x by `baseline` metres. That is
    the whole camera model here -- see the module docstring on why there is no
    distortion in it.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    @classmethod
    def from_stereo(cls, calibration):
        """Read them off a `StereoCalibration`.

        Only touches that object's own API, so this module still imports no
        OpenCV; the calibration does its own rectification internally.
        """
        _, _, P1, _, _ = calibration.rectify()
        return cls(
            fx=float(P1[0, 0]),
            fy=float(P1[1, 1]),
            cx=float(P1[0, 2]),
            cy=float(P1[1, 2]),
            baseline=float(calibration.baseline),
        )

    @property
    def K(self):
        """The shared 3x3 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def project(self, points, eye="left"):
        """Project camera-frame points to rectified pixels.

        Args:
            points: An (N, 3) array of metres in the rectified left camera
                frame -- x right, y down, z forward.
            eye: "left" or "right". The right eye is the same camera shifted
                along x, so a point's two projections differ only in u.

        Returns:
            An (N, 2) array of pixels.
        """
        if eye not in ("left", "right"):
            raise HandEyeError(f"eye must be 'left' or 'right', not {eye!r}")
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if eye == "right":
            points = points - np.array([self.baseline, 0.0, 0.0])
        z = points[:, 2]
        if np.any(z <= 0):
            # Behind the camera there is no projection, and letting the divide
            # run would hand back a plausible-looking pixel for a point the
            # camera cannot see.
            raise HandEyeError(
                f"{int(np.sum(z <= 0))} of {len(points)} points are at or "
                "behind the camera plane, so they do not project"
            )
        return np.stack(
            [self.fx * points[:, 0] / z + self.cx, self.fy * points[:, 1] / z + self.cy],
            axis=1,
        )

    def triangulate(self, left, right):
        """Rectified pixel pairs to camera-frame metres, `(N, 2)` and `(N, 2)` in.

        The rectified inverse of `project`: matching points share a row, so the
        disparity is the difference in u alone.
        """
        left = np.asarray(left, dtype=float).reshape(-1, 2)
        right = np.asarray(right, dtype=float).reshape(-1, 2)
        if len(left) != len(right):
            raise HandEyeError(
                f"got {len(left)} left and {len(right)} right points -- these "
                "must already be matched up"
            )
        disparity = left[:, 0] - right[:, 0]
        if np.any(disparity <= 0):
            raise HandEyeError(
                f"{int(np.sum(disparity <= 0))} of {len(left)} points have "
                "non-positive disparity, so they cannot be triangulated -- the "
                "two eyes are probably swapped"
            )
        z = self.fx * self.baseline / disparity
        return np.stack(
            [
                (left[:, 0] - self.cx) * z / self.fx,
                (left[:, 1] - self.cy) * z / self.fy,
                z,
            ],
            axis=1,
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclasses.dataclass
class Observation:
    """What one arm pose contributed: the board seen in both eyes.

    `ids` are ChArUco corner ids, so `ids[k]` indexes
    `BoardSpec.object_points()` directly and the left/right correspondence is
    exact rather than matched geometrically. Only corners found in *both* eyes
    belong here -- see `fit_board_pose` for why.
    """

    flange: np.ndarray  # T_base_flange, 4x4, from forward kinematics
    ids: np.ndarray  # (N,) int
    left: np.ndarray  # (N, 2) rectified pixels
    right: np.ndarray  # (N, 2) rectified pixels

    def __post_init__(self):
        self.flange = _as_pose(self.flange, "flange pose")
        self.ids = np.asarray(self.ids, dtype=int).reshape(-1)
        self.left = np.asarray(self.left, dtype=float).reshape(-1, 2)
        self.right = np.asarray(self.right, dtype=float).reshape(-1, 2)
        if not len(self.ids) == len(self.left) == len(self.right):
            raise HandEyeError(
                f"an observation has {len(self.ids)} ids but {len(self.left)} "
                f"left and {len(self.right)} right corners"
            )

    def __len__(self):
        return len(self.ids)


@dataclasses.dataclass(frozen=True)
class TagReference:
    """What the table tag looked like while the calibration was known good.

    Rectified pixels, in TL TR BR BL order, at `resolution`. `intrinsics` are
    this resolution's rather than the calibration's, because the two need not
    match: a calibration may be made in HD, and a tag this small wants 2K.
    `length` is what the tag was declared to be. It is kept for the record and
    plays no part in the arithmetic.
    """

    tag_id: int
    dictionary: str
    length: float
    resolution: str
    intrinsics: Intrinsics
    left: tuple
    right: tuple
    created: str

    @property
    def points(self):
        """The four corners in the camera frame at the time, `(4, 3)` metres."""
        return self.intrinsics.triangulate(self.left, self.right)

    def to_dict(self):
        data = dataclasses.asdict(self)
        data["left"] = np.asarray(self.left).tolist()
        data["right"] = np.asarray(self.right).tolist()
        return data

    @classmethod
    def from_dict(cls, data):
        fields = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in data.items() if key in fields}
        values["intrinsics"] = Intrinsics.from_dict(data["intrinsics"])
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class CameraShift:
    """How far the camera has moved since a `TagReference` was recorded.

    `T_old_new` is the camera's new pose in its old frame, so
    `T_base_cam @ T_old_new` is the corrected calibration. The two errors are
    the one-sigma position error the fit expects, at the tag and `ERROR_RADIUS`
    from it, in metres. They are a floor rather than a promise: corner bias
    that changes with the view does not show up as scatter, and on renders of a
    small tag on poor images the real error is two to four times this.
    """

    T_old_new: list
    rotation: float  # degrees
    bearing: float  # degrees, how far the tag's centre has moved in the picture
    range_change: float  # m, positive for further from the tag
    rms_px: float  # how well the four corners fit, left eye
    error_at_tag: float
    error_away: float
    measured: str

    @property
    def moved(self):
        return (
            self.bearing > MOVED_BEARING
            or self.rotation > MOVED_ROTATION
            or abs(self.range_change) > MOVED_RANGE
        )

    def describe(self):
        return f"{self.rotation:.2f} deg, {self.range_change * 1000:+.0f} mm in range"

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})


def fit_camera_shift(reference, left, right, noise_px=0.0):
    """How the camera has moved, from the tag's corners now. Returns a `CameraShift`.

    `left` and `right` are the four corners in the same order and at the same
    resolution as the reference. The module docstring has the reasoning; what
    is fitted is the 6-DOF motion `D` such that the reference corners, seen from
    the moved camera, land where they are now seen.

    `noise_px` is how uncertain those corners really are, from `capture_tag`. It
    sets how far the errors reported can be believed, and how much the fit leans
    on the prior: on a poor image the tag's own tilt deserves less trust.
    """
    sigma_px = max(SHIFT_SIGMA_PX, float(noise_px))
    sigma_disparity = max(SHIFT_SIGMA_DISPARITY, float(noise_px) * np.sqrt(2.0))
    intrinsics = reference.intrinsics
    left = np.asarray(left, dtype=float).reshape(4, 2)
    right = np.asarray(right, dtype=float).reshape(4, 2)
    points = reference.points
    seen_disparity = left[:, 0] - right[:, 0]

    centre = points.mean(axis=0)
    ray = centre / np.linalg.norm(centre)
    across = np.eye(3) - np.outer(ray, ray)
    focal_baseline = intrinsics.fx * intrinsics.baseline

    def in_new_camera(x, at):
        # D^-1 p, with `at` in the old camera frame.
        return (at - x[3:]) @ Rotation.from_rotvec(x[:3]).as_matrix()

    def corner_error(x):
        return intrinsics.project(in_new_camera(x, points), "left") - left

    def residuals(x):
        depth = in_new_camera(x, points)[:, 2]
        return np.concatenate([
            corner_error(x).ravel() / sigma_px,
            (focal_baseline / depth - seen_disparity) / sigma_disparity,
            across @ x[3:] / SHIFT_SIGMA_T,
        ])

    result = least_squares(residuals, np.zeros(6), xtol=1e-12, ftol=1e-12)
    x = result.x
    covariance = np.linalg.inv(result.jac.T @ result.jac)

    def position_error(c):
        # D c is about c + w x c + t, so that is what the covariance maps through.
        skew = np.array(
            [[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]]
        )
        jacobian = np.hstack([-skew, np.eye(3)])
        return float(np.sqrt(np.trace(jacobian @ covariance @ jacobian.T)))

    _, _, frame = np.linalg.svd(ray.reshape(1, 3))
    away = [
        centre + ERROR_RADIUS * (np.cos(angle) * frame[1] + np.sin(angle) * frame[2])
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ]

    centre_now = in_new_camera(x, centre)
    return CameraShift(
        T_old_new=pose(x[:3], x[3:]).tolist(),
        rotation=float(np.degrees(np.linalg.norm(x[:3]))),
        bearing=float(np.degrees(np.arccos(np.clip(
            ray @ centre_now / np.linalg.norm(centre_now), -1.0, 1.0
        )))),
        range_change=float(np.linalg.norm(centre_now) - np.linalg.norm(centre)),
        rms_px=float(np.sqrt(np.mean(np.sum(corner_error(x) ** 2, axis=1)))),
        error_at_tag=position_error(centre),
        error_away=max(position_error(c) for c in away),
        measured=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# ----------------------------------------------------------------------
# Per-pose board pose
# ----------------------------------------------------------------------


def fit_board_pose(observation, intrinsics, object_points):
    """Where the board is in the camera frame, for one pose: `T_cam_board`.

    Triangulates the corners and fits the board's known geometry onto the
    result -- an orthogonal Procrustes problem, solved in closed form.

    This is a stereo fit rather than a PnP for one reason: a planar target
    admits two PnP solutions mirrored through the board plane, and telling them
    apart from a single view is exactly as hard as the noise makes it. With two
    eyes there is no ambiguity to resolve -- the corners have metric depth, and
    only one of the two candidate orientations puts them where they were
    measured. That is also why `Observation` keeps only corners seen in both
    eyes: a corner in one eye adds nothing a stereo fit can use, and the whole
    board is in both eyes for any pose worth calibrating from.

    Accurate to a few tenths of a degree, which is a starting point for
    `refine` rather than an answer.
    """
    measured = intrinsics.triangulate(observation.left, observation.right)
    known = np.asarray(object_points, dtype=float)[observation.ids]
    if len(measured) < 3:
        raise HandEyeError(
            f"a board pose needs at least 3 corners, got {len(measured)}"
        )

    centre_measured = measured.mean(axis=0)
    centre_known = known.mean(axis=0)
    rotation = _nearest_rotation(
        (measured - centre_measured).T @ (known - centre_known)
    )
    return pose(rotation, centre_measured - rotation @ centre_known)


# ----------------------------------------------------------------------
# The closed form
# ----------------------------------------------------------------------


def solve(flange_poses, board_poses, min_rotation=0.01):
    """Solve `X A(i) = B(i) Y` in closed form. Returns `(T_base_cam, T_flange_board)`.

    Park and Martin's method, applied to relative motions. Taking two poses at
    a time cancels both unknowns out of one side:

        A(i) A(j)^-1  =  X^-1 (B(i) B(j)^-1) X

    which is the classic `M = X^-1 N X`, and on the rotations alone it says
    that `X` maps one log-map axis onto the other. Stacking the outer products
    of those axis pairs over every pair of poses and taking the nearest
    rotation gives `X`'s rotation in one SVD, with no initial guess and no
    iteration.

    With both rotations known, the translations are linear: `X A = B Y`
    expanded on the fourth column is `Rx tA + tx = RB ty + tB`, which is three
    equations per pose in the six unknowns of `tx` and `ty` together. Every
    pose goes into one least-squares rather than being averaged afterwards.

    Args:
        flange_poses: `T_base_flange` per pose, from forward kinematics.
        board_poses: `T_cam_board` per pose, from the camera.
        min_rotation: radians of relative rotation below which a pair of poses
            is skipped. A pair that barely rotated has a log-map axis made of
            noise, and it would be weighted like any other.

    Returns:
        `(T_base_cam, T_flange_board)`, both 4x4.
    """
    A = [_as_pose(T, "board pose") for T in board_poses]
    B = [_as_pose(T, "flange pose") for T in flange_poses]
    if len(A) != len(B):
        raise HandEyeError(f"got {len(B)} flange poses and {len(A)} board poses")
    if len(A) < 3:
        raise HandEyeError(
            f"hand-eye needs at least 3 poses and really wants more like 15, "
            f"got {len(A)}"
        )

    scatter = np.zeros((3, 3))
    used = 0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            axis_cam = _rotvec((A[i] @ invert(A[j]))[:3, :3])
            axis_base = _rotvec((B[i] @ invert(B[j]))[:3, :3])
            if np.linalg.norm(axis_cam) < min_rotation:
                continue
            # Rx maps the camera-side axis onto the base-side one.
            scatter += np.outer(axis_base, axis_cam)
            used += 1
    if used < 2:
        raise HandEyeError(
            f"only {used} of the {len(A) * (len(A) - 1) // 2} pose pairs rotate "
            f"by more than {np.degrees(min_rotation):.1f} deg, which is not "
            "enough to orient the camera. The poses need to differ in "
            "*rotation*, about more than one axis -- moving the arm around "
            "without turning the wrist leaves this unsolvable."
        )
    R_cam = _nearest_rotation(scatter)

    rows, rhs = [], []
    for board, flange in zip(A, B):
        rows.append(np.hstack([np.eye(3), -flange[:3, :3]]))
        rhs.append(flange[:3, 3] - R_cam @ board[:3, 3])
    translations, *_ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(rhs), rcond=None
    )

    # Y's rotation is over-determined the same way: every pose offers an
    # estimate, so sum them and project back onto SO(3) once.
    R_board = _nearest_rotation(
        sum(flange[:3, :3].T @ R_cam @ board[:3, :3] for board, flange in zip(A, B))
    )
    return (
        pose(R_cam, translations[:3]),
        pose(R_board, translations[3:]),
    )


# ----------------------------------------------------------------------
# The refinement
# ----------------------------------------------------------------------


def _pack(T_base_cam, T_flange_board):
    return np.concatenate(
        [
            _rotvec(T_base_cam[:3, :3]),
            T_base_cam[:3, 3],
            _rotvec(T_flange_board[:3, :3]),
            T_flange_board[:3, 3],
        ]
    )


def _unpack(parameters):
    return (
        pose(parameters[0:3], parameters[3:6]),
        pose(parameters[6:9], parameters[9:12]),
    )


def _residuals(parameters, observations, intrinsics, object_points):
    T_base_cam, T_flange_board = _unpack(parameters)
    T_cam_base = invert(T_base_cam)
    out = []
    for observation in observations:
        # The whole point of the parameterization: the board's pose is not
        # fitted, it is implied by the two transforms and this pose's FK.
        T_cam_board = T_cam_base @ observation.flange @ T_flange_board
        corners = object_points[observation.ids]
        in_camera = corners @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
        out.append((intrinsics.project(in_camera, "left") - observation.left).ravel())
        out.append((intrinsics.project(in_camera, "right") - observation.right).ravel())
    return np.concatenate(out)


def reprojection_errors(T_base_cam, T_flange_board, observations, intrinsics,
                        object_points):
    """Per-corner reprojection error in pixels, as an `(M,)` array of distances.

    What `refine` minimised, read back as something a TA can judge: these are
    pixel distances, two per corner per eye collapsed into one.
    """
    flat = _residuals(
        _pack(_as_pose(T_base_cam), _as_pose(T_flange_board)),
        list(observations),
        intrinsics,
        np.asarray(object_points, dtype=float),
    )
    return np.linalg.norm(flat.reshape(-1, 2), axis=1)


def refine(observations, intrinsics, object_points, T_base_cam, T_flange_board,
           knee=OUTLIER_KNEE):
    """Fit all 12 parameters against every corner in both eyes.

    The accuracy-determining step. `solve` gets within a few millimetres using
    only the distilled per-pose board poses; this uses the corners themselves,
    which is where the information actually is, and takes it to tenths of a
    millimetre.

    Args:
        observations: the `Observation` list.
        intrinsics: rectified `Intrinsics`.
        object_points: the board's corners, `BoardSpec.object_points()`.
        T_base_cam: starting guess, from `solve`.
        T_flange_board: starting guess, from `solve`.
        knee: Huber's knee in pixels; see `OUTLIER_KNEE`.

    Returns:
        `(T_base_cam, T_flange_board, stats)`, where `stats` is the dict that
        goes into the stored file.
    """
    observations = list(observations)
    if not observations:
        raise HandEyeError("no observations to refine against")
    object_points = np.asarray(object_points, dtype=float)

    start = _pack(_as_pose(T_base_cam), _as_pose(T_flange_board))
    args = (observations, intrinsics, object_points)

    before = np.linalg.norm(_residuals(start, *args).reshape(-1, 2), axis=1)
    result = least_squares(
        _residuals,
        start,
        args=args,
        loss="huber",
        f_scale=knee,
        xtol=1e-12,
        ftol=1e-12,
    )
    if not result.success and result.status <= 0:
        raise HandEyeError(
            f"the refinement did not converge ({result.message}). The usual "
            "cause is a pose whose corners were mis-detected; check the "
            "per-pose errors."
        )
    T_base_cam, T_flange_board = _unpack(result.x)

    after = reprojection_errors(
        T_base_cam, T_flange_board, observations, intrinsics, object_points
    )
    # Per pose, which is what finds the one bad pose. Each corner contributes
    # one error per eye, hence the doubling.
    per_pose, at = [], 0
    for observation in observations:
        count = 2 * len(observation)
        per_pose.append(float(np.sqrt(np.mean(after[at:at + count] ** 2))))
        at += count

    stats = {
        "poses": len(observations),
        "corners": int(sum(len(o) for o in observations)),
        "rms_px": float(np.sqrt(np.mean(after**2))),
        "max_px": float(np.max(after)),
        "rms_px_before_refine": float(np.sqrt(np.mean(before**2))),
        "per_pose_rms_px": per_pose,
        "outlier_knee_px": float(knee),
    }
    return T_base_cam, T_flange_board, stats


def leave_one_out(observations, intrinsics, object_points, T_base_cam,
                  T_flange_board, knee=OUTLIER_KNEE):
    """How far the camera position moves when each pose is dropped, in metres.

    The number that says whether a calibration is trustworthy rather than
    merely self-consistent. A low reprojection error only means the 12
    parameters explain the corners that were collected; it says nothing about
    whether a different set of poses would have given a different answer. This
    refits once per dropped pose and reports the spread, so a pose set too
    narrow to pin the camera down shows up as millimetres or centimetres of
    wander even while the residuals look excellent.

    Returns `(spread, positions)`: the largest distance from the mean, and the
    `(n_poses, 3)` positions themselves.
    """
    observations = list(observations)
    if len(observations) < 4:
        raise HandEyeError(
            f"leaving one out needs at least 4 poses, got {len(observations)}"
        )
    object_points = np.asarray(object_points, dtype=float)

    positions = []
    for dropped in range(len(observations)):
        kept = observations[:dropped] + observations[dropped + 1:]
        result = least_squares(
            _residuals,
            _pack(T_base_cam, T_flange_board),
            args=(kept, intrinsics, object_points),
            loss="huber",
            f_scale=knee,
        )
        positions.append(_unpack(result.x)[0][:3, 3])

    positions = np.asarray(positions)
    spread = float(np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
    return spread, positions


# ----------------------------------------------------------------------
# The result, and the file it lives in
# ----------------------------------------------------------------------


class HandEye:
    """One bench's camera-to-base calibration.

    `T_base_cam` is the whole point of it: a point in the rectified left
    camera frame, left-multiplied by this, is the same point in the robot's
    base frame. `points_to_base` is that multiplication with the bookkeeping
    done.

    Everything else is here so the file can be audited years later: which
    camera, at what resolution, against which board, from how many poses, and
    how well it fitted.

    `shift` is None for the calibration as filed. When it is set the camera has
    been knocked, `T_base_cam` is the corrected transform, and
    `T_base_cam_filed` is what the TA measured.
    """

    def __init__(self, T_base_cam, T_flange_board, intrinsics, board, serial=None,
                 resolution=None, station=None, stats=None, created=None,
                 source=None, tag=None):
        self.T_base_cam = _as_pose(T_base_cam, "T_base_cam")
        self.T_base_cam_filed = self.T_base_cam
        self.tag = tag
        self.shift = None
        self.T_flange_board = _as_pose(T_flange_board, "T_flange_board")
        self.intrinsics = intrinsics
        self.board = board
        self.serial = None if serial is None else str(serial)
        self.resolution = resolution
        self.station = station
        self.stats = dict(stats or {})
        self.created = created or time.strftime("%Y-%m-%dT%H:%M:%S")
        # Where `load` read it from, so a surprising answer can be traced to
        # the file that gave it. Not stored in the file itself.
        self.source = source

    # ------------------------------------------------------------------
    # Using it
    # ------------------------------------------------------------------

    @property
    def T_cam_base(self):
        """The other direction: base frame into camera frame."""
        return invert(self.T_base_cam)

    @property
    def camera_position(self):
        """Where the camera is, in base coordinates, in metres."""
        return self.T_base_cam[:3, 3].copy()

    @property
    def camera_forward(self):
        """Unit vector the camera looks along, in base coordinates.

        The optical axis is +z in the camera frame, so this is the third
        column of the rotation.
        """
        return self.T_base_cam[:3, 2].copy()

    def points_to_base(self, points):
        """Camera-frame metres to base-frame metres. `(N, 3)` in, `(N, 3)` out.

        What a student calls on the output of `StereoCalibration.triangulate`.
        A single `(3,)` point is accepted and comes back as `(1, 3)`.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return points @ self.T_base_cam[:3, :3].T + self.T_base_cam[:3, 3]

    def points_to_camera(self, points):
        """Base-frame metres to camera-frame metres, the reverse."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        T = self.T_cam_base
        return points @ T[:3, :3].T + T[:3, 3]

    def project(self, points, eye="left"):
        """Where a base-frame point lands in a rectified image, in pixels.

        For drawing the robot's own coordinates back onto what the camera sees,
        which is the quickest way to see whether a calibration is sane.
        """
        return self.intrinsics.project(self.points_to_camera(points), eye=eye)

    def board_pose(self, flange_pose):
        """Where the board should be in the camera frame at a given flange pose.

        `X^-1 B Y`, to compare against what the camera actually sees.
        """
        return self.T_cam_base @ _as_pose(flange_pose, "flange pose") @ self.T_flange_board

    def corrected(self, shift):
        """A copy for a camera that has moved by `shift` since this was filed."""
        out = copy.copy(self)
        out.T_base_cam = self.T_base_cam_filed @ np.asarray(shift.T_old_new)
        out.shift = shift
        return out

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def to_dict(self):
        # The filed transform even on a corrected copy: a correction is a
        # stopgap with its own file, and must not be saved as if it were measured.
        return {
            "format": _FORMAT,
            "created": self.created,
            "serial": self.serial,
            "resolution": self.resolution,
            "station": self.station,
            "tag": None if self.tag is None else self.tag.to_dict(),
            "T_base_cam": self.T_base_cam_filed.tolist(),
            "T_flange_board": self.T_flange_board.tolist(),
            "intrinsics": self.intrinsics.to_dict(),
            "board": self.board.to_dict(),
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data, source=None):
        if data.get("format") != _FORMAT:
            raise HandEyeError(
                f"this is a format {data.get('format')!r} calibration and this "
                f"version of xarm7_lib reads format {_FORMAT} -- it was written "
                "by a different version of the library, so recalibrate rather "
                "than guessing"
            )
        try:
            return cls(
                T_base_cam=np.asarray(data["T_base_cam"], dtype=float),
                T_flange_board=np.asarray(data["T_flange_board"], dtype=float),
                intrinsics=Intrinsics.from_dict(data["intrinsics"]),
                board=BoardSpec.from_dict(data["board"]),
                serial=data.get("serial"),
                resolution=data.get("resolution"),
                station=data.get("station"),
                stats=data.get("stats"),
                created=data.get("created"),
                source=source,
                tag=TagReference.from_dict(data["tag"]) if data.get("tag") else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HandEyeError(f"calibration is missing or malformed: {error}") from error

    def save(self, path):
        """Write it as JSON, creating the directory if it isn't there.

        Returns the path written, which is worth printing: the default is
        inside the installation and not where a TA is standing.
        """
        path = pathlib.Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        except OSError as error:
            raise HandEyeError(f"could not write {path}: {error}") from error
        return path

    @classmethod
    def load(cls, path):
        """Read one back from an explicit path."""
        path = pathlib.Path(path)
        try:
            data = json.loads(path.read_text())
        except OSError as error:
            raise HandEyeError(f"could not read {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise HandEyeError(f"{path} is not valid JSON: {error}") from error
        return cls.from_dict(data, source=path)

    def __repr__(self):
        serial = f"SN{self.serial}" if self.serial else "unknown serial"
        x, y, z = self.camera_position
        rms = self.stats.get("rms_px")
        quality = f" {rms:.2f}px" if rms is not None else ""
        moved = "" if self.shift is None else " CORRECTED, camera has moved"
        return (
            f"<HandEye {serial} camera at ({x:.3f}, {y:.3f}, {z:.3f}) m"
            f"{quality}{moved}>"
        )


def filename(serial):
    """What a camera's calibration is called. Keyed by serial, so stations don't collide."""
    return f"SN{serial}.json"


def search_paths(serial):
    """Every place `load_handeye` looks, in the order it looks.

    Exposed because the interesting failure is "it found the wrong one", and
    answering that means seeing the list.
    """
    paths = []
    override = os.environ.get(HANDEYE_ENV, "").strip()
    if override:
        paths.append(pathlib.Path(override).expanduser())
    if serial is not None:
        paths.append(USER_DIR / filename(serial))
        paths.append(INSTALLED_DIR / filename(serial))
    return paths


def warn_moved(handeye, shift):
    """The message the TA is meant to see, printed when a correction is put in use.

    Printed by whatever is watching the table tag -- the capstone's
    `LightsaberDetector`, which streams the camera and so is the one thing that
    can see the tag.
    """
    bar = "=" * 72
    print(
        f"[xarm7_lib] WARNING: THE CAMERA HAS MOVED\n{bar}\n"
        f"  Camera SN{handeye.serial} is not where it was when it was calibrated "
        f"on {handeye.created}.\n"
        f"  Measured from the table tag at {shift.measured}: moved "
        f"{shift.describe()}.\n\n"
        "  Students: you can keep working. This run uses a corrected transform.\n"
        f"  Expect errors of roughly {shift.error_at_tag * 1000:.0f} mm near the tag "
        f"and {shift.error_away * 1000:.0f} mm {ERROR_RADIUS:.1f} m away\n"
        "  from it, and possibly a few times that.\n\n"
        "  TA: please redo the hand-eye calibration on this bench:\n"
        "      python -m xarm7_lib.calibrate_handeye run\n"
        "  If it was the tag that got moved rather than the camera, this\n"
        "  correction is wrong and the calibration still needs redoing.\n"
        f"{bar}",
        file=sys.stderr,
        flush=True,
    )


def load_handeye(serial=None):
    """This bench's camera-to-base calibration.

    The one call a student makes. A TA has already run
    `python -m xarm7_lib.calibrate_handeye` on this station and filed the
    result inside the installation, so there is no path to pass and nothing to
    configure:

        handeye = load_handeye()
        base_points = handeye.points_to_base(camera_points)

    Looked up in three places, first hit winning: `$XARM7_HANDEYE` if it is
    set, then `~/.config/xarm7_lib/handeye/`, then the installation. The last
    is the shared one. The first two outrank it so a TA can try a new
    calibration without disturbing the class's -- and the middle one is where
    `calibrate_handeye run` writes by default, before a `sudo mv` promotes it
    into the installation.

    This is the calibration as filed. It never opens the camera and never
    corrects for a knocked one: the table tag recorded with it (`tag`) is
    checked by whatever streams the camera, which in the capstone is
    `LightsaberDetector` -- see `fit_camera_shift` and `HandEye.corrected`.

    Args:
        serial: Which camera's calibration. Read from the attached camera when
            omitted -- which needs the native shim, so pass it explicitly on a
            machine that has none. It is printed on the camera's back panel.

    Returns:
        A `HandEye`.

    Raises:
        HandEyeError: There is no calibration.
    """
    if serial is None:
        # `$XARM7_HANDEYE` names a file outright, so it needs no serial and so
        # no camera. Checked before anything is probed, because the point of it
        # is to work on a machine that has no camera to probe -- a TA trying a
        # calibration on a laptop, or a test with a synthetic one.
        override = os.environ.get(HANDEYE_ENV, "").strip()
        if override:
            path = pathlib.Path(override).expanduser()
            if path.is_file():
                return HandEye.load(path)

        # Imported here and not at the top of the module: finding the attached
        # camera needs OpenCV and the shim, and loading a calibration for a
        # named camera needs neither.
        try:
            from .camera import find_device
        except ImportError as error:
            raise HandEyeError(
                "finding the attached camera needs the camera support, whose "
                "dependencies are an optional extra. Either install them with "
                "`pip install -e '.[camera]'` or pass serial= -- it is printed "
                f"on the camera's back panel. ({error})"
            ) from error
        try:
            _, serial = find_device()
        except Exception as error:
            # Whatever went wrong looking for the camera, what the caller asked
            # for was a calibration, so that is the failure they should get --
            # and the two ways out are the same either way.
            raise HandEyeError(
                "no camera could be found, so there is no serial number to look "
                "a calibration up by. Pass serial= -- it is printed on the "
                f"camera's back panel -- or set ${HANDEYE_ENV} to a calibration "
                f"file. ({error})"
            ) from error
        if serial is None:
            raise HandEyeError(
                "the attached camera's serial number could not be read, so its "
                "calibration cannot be looked up. Pass serial= -- it is printed "
                "on the camera's back panel."
            )

    return find_filed(serial)


def find_filed(serial):
    """A camera's calibration as filed, by serial, without probing for one."""
    serial = str(serial)
    paths = search_paths(serial)
    for path in paths:
        if path.is_file():
            return HandEye.load(path)

    looked = "\n    ".join(str(path) for path in paths)
    raise HandEyeError(
        f"no hand-eye calibration for camera SN{serial}. Looked in:\n"
        f"    {looked}\n"
        "This is calibrated once per bench by a TA, not per student -- if it "
        "is missing, the station needs `python -m xarm7_lib.calibrate_handeye` "
        "run on it by someone who can write to the installation."
    )

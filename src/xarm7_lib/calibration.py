"""Factory stereo calibration for a ZED camera, without the ZED SDK.

Every camera is calibrated individually at the factory, so these parameters
belong to one physical unit and are looked up by its serial number. Stereolabs
keeps that per-serial calibration on a public endpoint, which is what this
module reads:

    http://calib.stereolabs.com/?SN=<serial>

The result is a plain INI file -- a `[LEFT_CAM_<res>]` and `[RIGHT_CAM_<res>]`
section per video mode, plus one `[STEREO]` section -- so neither the SDK nor an
NVIDIA GPU is involved in getting it. That matters here: the SDK needs CUDA to
open a camera at all, and `getCameraInformation()` needs an opened camera, so on
a machine without an NVIDIA GPU the SDK cannot hand over the calibration even
though the data itself has nothing to do with the GPU.

Three sources are tried in order, the same order the SDK itself uses:

    1. `/usr/local/zed/settings/SN<serial>.conf`, where the SDK caches it and
       where Stereolabs' calibration tool writes a *custom* calibration.
    2. a local cache under `~/.cache/xarm7_lib/zed`, written on first download.
    3. the endpoint above.

The ordering is not arbitrary. A camera that has been recalibrated -- knocked
out of alignment and corrected with Stereolabs' tool -- has a calibration that
exists only as a local file on whichever machine ran the tool. The endpoint
still serves the original factory numbers and knows nothing about the
recalibration, so a local file always wins. If this camera was recalibrated
elsewhere, copy its `SN<serial>.conf` into one of the first two locations;
downloading will quietly give the wrong answer otherwise.

The camera also carries its calibration on an onboard EEPROM, but reading that
is an SDK 5.3+ feature and so needs CUDA like the rest of the SDK. The endpoint
is the way in without one.
"""

import configparser
import os
import pathlib
import urllib.request

import cv2
import numpy as np

CALIB_URL = "http://calib.stereolabs.com/?SN={serial}"

# Per-eye image size for each of the camera's video modes. These names are also
# the suffixes on the conf file's sections and on its per-resolution stereo
# keys, so this table is what picks which part of the file to read.
RESOLUTIONS = {
    "2K": (2208, 1242),
    "FHD": (1920, 1080),
    "HD": (1280, 720),
    "VGA": (672, 376),
}

_SDK_SETTINGS = pathlib.Path("/usr/local/zed/settings")
_CACHE = (
    pathlib.Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    / "xarm7_lib"
    / "zed"
)

_DOWNLOAD_TIMEOUT = 10.0

# Radial and tangential terms, in the order cv2 expects them.
_DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")


class CalibrationError(RuntimeError):
    """This camera's calibration could not be found, read, or made sense of."""


def find_conf(serial, download=True):
    """Return the path to one camera's conf file, fetching it if it is not local.

    Local copies take precedence over the download, which is the point -- see
    the module docstring for why.

    Args:
        serial: The camera's serial number, as printed on its back panel.
        download: If False, fail rather than reaching for the network.

    Returns:
        A `pathlib.Path` to the conf file.
    """
    candidates = (_SDK_SETTINGS / f"SN{serial}.conf", _CACHE / f"SN{serial}.conf")
    for path in candidates:
        if path.is_file():
            return path

    if not download:
        locations = " or ".join(str(path.parent) for path in candidates)
        raise CalibrationError(f"no calibration for SN{serial} in {locations}")

    url = CALIB_URL.format(serial=serial)
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
            text = response.read().decode("utf-8", "replace")
    except OSError as error:
        raise CalibrationError(f"could not fetch {url}: {error}") from error

    # The endpoint answers 200 with an error page for a serial it does not
    # know, so the only way to tell a hit from a miss is to look inside.
    if "[STEREO]" not in text:
        raise CalibrationError(
            f"{url} returned something that is not a calibration file -- check "
            f"that {serial} is really this camera's serial number"
        )

    destination = _CACHE / f"SN{serial}.conf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def _intrinsics(conf, section):
    if section not in conf:
        raise CalibrationError(f"conf file has no [{section}] section")
    values = conf[section]
    try:
        K = np.array(
            [
                [float(values["fx"]), 0.0, float(values["cx"])],
                [0.0, float(values["fy"]), float(values["cy"])],
                [0.0, 0.0, 1.0],
            ]
        )
        D = np.array([float(values.get(key, 0.0)) for key in _DISTORTION_KEYS])
    except (KeyError, ValueError) as error:
        raise CalibrationError(f"[{section}] is missing or malformed: {error}") from error
    return K, D


def _extrinsics(conf, resolution):
    if "STEREO" not in conf:
        raise CalibrationError("conf file has no [STEREO] section")
    stereo = conf["STEREO"]

    def value(key, default=0.0):
        # Most keys are written per resolution (CV_HD, RX_HD); Baseline, and
        # often TY and TZ, are written once for the whole file.
        for name in (f"{key}_{resolution}", key):
            if name in stereo:
                return float(stereo[name])
        return default

    # The right eye's pose relative to the left is stored as a rotation vector
    # in (x, convergence about y, z) order, and a translation in millimetres.
    rotation, _ = cv2.Rodrigues(np.array([value("RX"), value("CV"), value("RZ")]))

    baseline = value("Baseline")
    if not baseline:
        raise CalibrationError("[STEREO] has no Baseline, so there is no stereo geometry")

    # cv2 wants the left camera's origin expressed in the right camera's frame.
    # The right eye sits at +x from the left, so these come out negative.
    # `rectify` checks the result, because this is the one convention in the
    # file that is easy to get backwards.
    translation = np.array([[-baseline], [-value("TY")], [-value("TZ")]]) / 1000.0
    return rotation, translation


class StereoCalibration:
    """One camera's stereo calibration at one resolution.

    Holds the per-eye intrinsics (`K1`/`D1`, `K2`/`D2`) and the stereo
    extrinsics (`R`, `T`, metres) as cv2 wants them, and derives the
    rectification from those on first use.

    Construct it with `from_serial` in normal use; `from_file` is there for a
    conf file you already have on disk.
    """

    def __init__(self, conf, resolution="HD", serial=None):
        if resolution not in RESOLUTIONS:
            raise CalibrationError(
                f"unknown resolution {resolution!r}, expected one of "
                f"{', '.join(RESOLUTIONS)}"
            )
        self.resolution = resolution
        self.serial = serial
        self.image_size = RESOLUTIONS[resolution]
        self.K1, self.D1 = _intrinsics(conf, f"LEFT_CAM_{resolution}")
        self.K2, self.D2 = _intrinsics(conf, f"RIGHT_CAM_{resolution}")
        self.R, self.T = _extrinsics(conf, resolution)
        self._rectified = None
        self._maps = None

    @classmethod
    def from_serial(cls, serial, resolution="HD", download=True):
        """Load the calibration belonging to the camera with this serial number."""
        path = find_conf(serial, download=download)
        return cls.from_file(path, resolution=resolution, serial=serial)

    @classmethod
    def from_file(cls, path, resolution="HD", serial=None):
        """Load a conf file from disk."""
        # strict=False: these files are machine-written and have been seen with
        # repeated keys, which the default parser refuses outright.
        conf = configparser.ConfigParser(strict=False)
        try:
            conf.read_string(pathlib.Path(path).read_text())
        except (OSError, configparser.Error) as error:
            raise CalibrationError(f"could not read {path}: {error}") from error
        return cls(conf, resolution=resolution, serial=serial)

    def rectify(self):
        """Return `(R1, R2, P1, P2, Q)` from `cv2.stereoRectify`, computed once."""
        if self._rectified is None:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                self.K1,
                self.D1,
                self.K2,
                self.D2,
                self.image_size,
                self.R,
                self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
            # For a rectification referenced to the left eye, P2[0, 3] is -f*B
            # and so negative. Positive means the translation picked up the
            # wrong sign and every depth computed from it would come out behind
            # the camera -- better to say so here than to hand back a mirrored
            # reconstruction that looks plausible until it is measured.
            if P2[0, 3] >= 0:
                raise CalibrationError(
                    "rectification put the right eye on the wrong side of the "
                    f"left (P2[0,3] = {P2[0, 3]:.1f}); the sign convention on "
                    "Baseline/TY/TZ does not match this conf file"
                )
            self._rectified = (R1, R2, P1, P2, Q)
        return self._rectified

    @property
    def focal_length(self):
        """Rectified focal length in pixels, shared by both eyes after rectification."""
        return self.rectify()[2][0, 0]

    @property
    def baseline(self):
        """Distance between the two eyes in metres, as rectification sees it."""
        _, _, P1, P2, _ = self.rectify()
        return -P2[0, 3] / P1[0, 0]

    def maps(self):
        """Return `((map1x, map1y), (map2x, map2y))` for `cv2.remap`, computed once.

        Only needed for whole-image rectification. For a handful of points,
        `undistort_points` does the same job without touching every pixel.
        """
        if self._maps is None:
            R1, R2, P1, P2, _ = self.rectify()
            self._maps = (
                cv2.initUndistortRectifyMap(
                    self.K1, self.D1, R1, P1, self.image_size, cv2.CV_32FC1
                ),
                cv2.initUndistortRectifyMap(
                    self.K2, self.D2, R2, P2, self.image_size, cv2.CV_32FC1
                ),
            )
        return self._maps

    def rectify_images(self, left, right):
        """Rectify a stereo pair, so that matching features share an image row."""
        (map1x, map1y), (map2x, map2y) = self.maps()
        return (
            cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR),
            cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR),
        )

    def undistort_points(self, points, eye="left"):
        """Move raw image points into rectified pixel coordinates.

        The cheap counterpart to `rectify_images`: rectifying a whole image is
        only needed so that dense matching can scan along rows, and if all you
        have is a few detections then this is arithmetic on those points
        instead of a remap of every pixel.

        Args:
            points: An (N, 2) array of pixel coordinates in the raw image.
            eye: "left" or "right", saying which image they came from.

        Returns:
            An (N, 2) array in that eye's rectified pixel coordinates.
        """
        if eye not in ("left", "right"):
            raise ValueError(f"eye must be 'left' or 'right', not {eye!r}")
        R1, R2, P1, P2, _ = self.rectify()
        if eye == "left":
            K, D, R, P = self.K1, self.D1, R1, P1
        else:
            K, D, R, P = self.K2, self.D2, R2, P2
        pixels = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pixels, K, D, R=R, P=P).reshape(-1, 2)

    def triangulate(self, left_points, right_points):
        """Triangulate matched detections into 3D points.

        Takes raw image coordinates -- the undistortion and rectification are
        applied here -- and returns metres in the rectified left camera frame,
        with x right, y down and z forward along the optical axis.

        Args:
            left_points: An (N, 2) array of pixel coordinates in the left image.
            right_points: The matching (N, 2) array from the right image.

        Returns:
            An (N, 3) array of positions in metres.
        """
        left = self.undistort_points(left_points, "left")
        right = self.undistort_points(right_points, "right")
        if len(left) != len(right):
            raise ValueError(
                f"got {len(left)} left points and {len(right)} right points -- "
                "triangulation needs them already matched up"
            )
        _, _, P1, P2, _ = self.rectify()
        homogeneous = cv2.triangulatePoints(P1, P2, left.T, right.T)
        return (homogeneous[:3] / homogeneous[3]).T

    def depth_from_disparity(self, disparity):
        """Convert rectified disparity in pixels to depth in metres."""
        disparity = np.asarray(disparity, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                disparity > 0, self.focal_length * self.baseline / disparity, np.nan
            )

    def __repr__(self):
        serial = f"SN{self.serial}" if self.serial else "unknown serial"
        return f"<StereoCalibration {serial} {self.resolution}>"

"""ZED stereo camera, with no ZED SDK and no CUDA.

Frames come from `zed-open-capture` through the shim in `native/`, and so does
the serial number that selects a calibration. The SDK is not involved anywhere:
it needs an NVIDIA GPU and the lab machine has none, but nothing about
capturing a ZED does.

Two things about this camera surprise people.

One frame holds both eyes. There is no mode that gives two streams -- a read
returns a single double-width image with the left eye in the left half. `read`
splits it; `read_raw` is there for when the frame as it arrived is what you
want.

The camera will not say which camera it is over USB. Its serial number lives in
a camera register reached by a vendor-specific request, not in the USB
descriptor, so udev has nothing to name the device after and several identical
cameras on a bench look alike. The serial is what selects a calibration, so it
is read from the camera itself through the shim -- see `resolve_serial`. That
means the shim is needed for a camera to find its own calibration, though
`serial=` still overrides it for a machine without one.

One trap is not this module's to fix but is worth knowing: a USB 2 port pins
the camera to its VGA mode however much is asked for, and sustained throughput
above roughly 200 MB/s has been seen to tear frames on this bench even on
USB 3 -- `light_stick.py --survey` measures that per mode.
"""

import dataclasses
import pathlib

import cv2
import numpy as np

from . import zed_controls
from .calibration import RESOLUTIONS, StereoCalibration

# What each mode runs at unless asked otherwise. These are the camera's maxima
# for the mode; the driver will accept lower rates.
DEFAULT_FPS = {"2K": 15, "FHD": 30, "HD": 60, "VGA": 100}

# Cameras are found through sysfs rather than by opening indices in turn: the
# card name says which nodes are ZEDs, and a UVC camera registers more than one
# node, of which only per-device index 0 carries video. The others return
# metadata, and opening one looks like a camera that produces nothing.
_SYSFS_V4L = pathlib.Path("/sys/class/video4linux")
_BY_PATH = pathlib.Path("/dev/v4l/by-path")
_CAPTURE_INDEX = "0"

# Asking a camera its serial means opening it, which only works while nothing
# is streaming it, so each answer is kept for the life of the process. Without
# this a second capture session would have to interrupt the first to identify it.
_SERIAL_CACHE = {}


class CameraError(RuntimeError):
    """The camera could not be found, opened, configured, or read from."""


@dataclasses.dataclass(frozen=True)
class ZedDevice:
    """One ZED capture node, and what is known about which camera it is."""

    path: pathlib.Path  # /dev/videoN, which is what cv2 opens
    node: str  # "videoN", the sysfs name
    port: str  # udev's by-path name: stable for the physical USB port
    usb_serial: str  # from the USB descriptor, which ZEDs leave empty

    def __repr__(self):
        return f"<ZedDevice {self.path} port={self.port or '?'}>"


def _sysfs_text(path, default=""):
    try:
        return path.read_text().strip()
    except OSError:
        return default


def _usb_serial(node):
    """The serial from the USB descriptor, or "".

    Empty on every ZED seen so far -- the serial the SDK reports lives in a
    camera register, not the descriptor. It is read anyway because it costs
    nothing, and a camera that does publish one needs no registration.
    """
    try:
        current = (_SYSFS_V4L / node / "device").resolve(strict=True)
    except OSError:
        return ""
    # Walk up to the USB device directory, the one that carries idVendor.
    for _ in range(6):
        if (current / "idVendor").is_file():
            return _sysfs_text(current / "serial")
        if current.parent == current:
            break
        current = current.parent
    return ""


def _port_name(node):
    """udev's by-path name for a node, which is stable per physical USB port."""
    if not _BY_PATH.is_dir():
        return ""
    device = pathlib.Path("/dev") / node
    for link in sorted(_BY_PATH.iterdir()):
        try:
            if link.resolve() == device.resolve():
                return link.name
        except OSError:
            continue
    return ""


def devices():
    """Every ZED capture node on this machine, as a list of `ZedDevice`."""
    if not _SYSFS_V4L.is_dir():
        raise CameraError(
            f"{_SYSFS_V4L} does not exist, so the kernel has registered no "
            "video device. Try unplugging and replugging the camera."
        )
    found = []
    for entry in sorted(_SYSFS_V4L.iterdir(), key=lambda path: path.name):
        if "ZED" not in _sysfs_text(entry / "name").upper():
            continue
        if _sysfs_text(entry / "index") != _CAPTURE_INDEX:
            continue
        found.append(
            ZedDevice(
                path=pathlib.Path("/dev") / entry.name,
                node=entry.name,
                port=_port_name(entry.name),
                usb_serial=_usb_serial(entry.name),
            )
        )
    return found


def resolve_serial(device, refresh=False):
    """Which camera is on this node, or None if it could not be asked.

    Prefers the USB descriptor on the off chance a camera publishes one, and
    otherwise asks the camera through the shim. Answers are cached per port, so
    a camera is opened once to identify it and later calls are free.

    Returns None rather than raising when the camera cannot be asked -- the
    shim is not built, or something is streaming it -- so that discovery still
    works and only the calibration lookup complains. `query_serial` is the
    version that reports why.

    Args:
        device: The `ZedDevice` to identify.
        refresh: Ask again rather than trusting the cache.
    """
    if device.usb_serial:
        return device.usb_serial

    key = device.port or device.node
    if refresh:
        _SERIAL_CACHE.pop(key, None)
    elif key in _SERIAL_CACHE:
        return _SERIAL_CACHE[key]

    try:
        serial = zed_controls.read_serial(device)
    except zed_controls.ControlsError:
        return None
    _SERIAL_CACHE[key] = serial
    return serial


def query_serial(device=None):
    """Ask a camera its own serial number, raising if it cannot be asked.

    What `resolve_serial` does without the cache and without swallowing the
    reason it failed. Nothing may be streaming the camera at the time.
    """
    return zed_controls.read_serial(device)


def _sole_device(attached=None):
    attached = devices() if attached is None else attached
    if not attached:
        raise CameraError(
            f"no ZED capture node under {_SYSFS_V4L} -- is a camera plugged in?"
        )
    if len(attached) > 1:
        listed = ", ".join(
            f"{device.path} (port {device.port or 'unknown'}, "
            f"serial {resolve_serial(device) or 'unknown'})"
            for device in attached
        )
        raise CameraError(
            f"{len(attached)} ZEDs are attached ({listed}) -- say which one "
            "with serial= or device="
        )
    return attached[0]


def find_device(serial=None):
    """Locate the capture node of an attached ZED.

    Args:
        serial: Which camera to find. Defaults to the only one attached.

    Returns:
        `(device, serial)`, where serial is None when nothing on this machine
        knows which camera it is.
    """
    attached = devices()
    if serial is None:
        device = _sole_device(attached)
        return device, resolve_serial(device)

    serial = str(serial)
    matches = [device for device in attached if resolve_serial(device) == serial]
    if len(matches) == 1:
        return matches[0], serial
    if not matches:
        listed = (
            ", ".join(
                f"{device.path}={resolve_serial(device) or 'unknown'}"
                for device in attached
            )
            or "none"
        )
        raise CameraError(
            f"no attached ZED answers to serial {serial} (attached: {listed}). "
            "A camera reads as unknown when the shim is not built or something "
            "else is streaming it."
        )
    raise CameraError(
        f"serial {serial} matched {len(matches)} of the attached cameras, which "
        "should be impossible -- two cameras cannot share a serial number."
    )


def _calibration_for(serial, resolution):
    """This camera's calibration, or why it cannot be found."""
    if serial is None:
        raise CameraError(
            "this camera's serial number could not be read, so its "
            "calibration cannot be looked up. The camera only gives it up "
            "through the zed-open-capture shim, so either that is not built "
            "(see native/README.md) or something was streaming the camera "
            "when it was asked. Passing serial= skips the question -- it is "
            "printed on the label on the camera's back panel."
        )
    return StereoCalibration.from_serial(serial, resolution)


class ZedOpenCapture:
    """A ZED stereo camera.

    Needs the native shim built -- see native/README.md.

    One handle streams and controls at once, so `set_exposure` and `set_gain`
    take effect without stopping the stream.

    Args:
        resolution: One of "2K", "FHD", "HD", "VGA", as a per-eye size.
        device: A `/dev/video*` path or an index; found automatically when
            omitted.
        serial: Which camera to open. Read from the camera itself otherwise.
        fps: Frame rate. Defaults to the mode's maximum.
        exposure: Applied once the stream is up, and changeable after.
        gain: The same.
    """

    def __init__(self, resolution="HD", device=None, serial=None, fps=None,
                 exposure=None, gain=None):
        if resolution not in RESOLUTIONS:
            raise CameraError(
                f"unknown resolution {resolution!r}, expected one of "
                f"{', '.join(RESOLUTIONS)}"
            )
        width, height = RESOLUTIONS[resolution]
        self.resolution = resolution
        self.size = (width, height)
        self.fps = DEFAULT_FPS[resolution] if fps is None else fps
        self.pixel_format = "YUYV"  # what the camera sends, before conversion
        self.timestamp = 0  # of the last frame, nanoseconds
        self.repeated = 0  # frames handed back twice, i.e. read faster than sent
        self._calibration = None
        self._last_frame_id = None

        if device is None:
            found = _sole_device() if serial is None else find_device(serial)[0]
            device, self.port = found.path, found.port
        else:
            self.port = ""
        self.device = device

        self._session = zed_controls.ZedControls(device, resolution, self.fps)
        self.serial = str(serial) if serial is not None else self._session.serial
        if exposure is not None:
            self._session.set_exposure(exposure)
        if gain is not None:
            self._session.set_gain(gain)
        self.exposure = self._session.exposure()
        self.gain = self._session.gain()

    @property
    def calibration(self):
        """This camera's factory calibration at the current resolution."""
        if self._calibration is None:
            self._calibration = _calibration_for(self.serial, self.resolution)
        return self._calibration

    def set_exposure(self, value):
        """Change exposure on both sensors, with the stream still running."""
        self._session.set_exposure(value)
        self.exposure = self._session.exposure()
        return self.exposure

    def set_gain(self, value):
        """Change gain on both sensors, with the stream still running."""
        self._session.set_gain(value)
        self.gain = self._session.gain()
        return self.gain

    def read_raw(self):
        """One frame as it arrives: both eyes side by side, converted to BGR."""
        frame, frame_id, timestamp = self._session.frame()
        if frame_id == self._last_frame_id:
            self.repeated += 1
        self._last_frame_id = frame_id
        self.timestamp = timestamp

        expected = (self.size[1], 2 * self.size[0])
        if frame.shape[:2] != expected:
            raise CameraError(
                f"the camera sent {frame.shape[1]}x{frame.shape[0]} where "
                f"{self.resolution} should be {expected[1]}x{expected[0]}"
            )
        # The conversion also copies the frame out of the session's reused
        # buffer, which the next grab overwrites.
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)

    def read(self):
        """Grab one frame and return it as `(left, right)`."""
        left, right = np.split(self.read_raw(), 2, axis=1)
        return np.ascontiguousarray(left), np.ascontiguousarray(right)

    def read_rectified(self):
        """Grab one frame and return the two eyes rectified."""
        return self.calibration.rectify_images(*self.read())

    def close(self):
        """Stop streaming and release the camera."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        serial = f"SN{self.serial}" if self.serial else "unknown serial"
        return f"<ZedOpenCapture {serial} {self.resolution} {self.size[0]}x{self.size[1]}>"

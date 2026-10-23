"""ZED stereo camera over plain UVC, with no ZED SDK and no CUDA.

A USB3 ZED is a UVC-compliant device, so its frames can be pulled straight off
`/dev/video*` with OpenCV. That is the whole reason this module exists: the ZED
SDK requires an NVIDIA GPU and the lab machine has none, but nothing about
*capturing* needs one. `calibration.py` gets the calibration the same way, so
between them no part of the pipeline wants the SDK.

Three things about this camera surprise people, and all three are handled here.

One frame holds both eyes. There is no mode that gives two streams -- a read
returns a single double-width image with the left eye in the left half. `read`
splits it; `read_raw` is there for when the frame as it arrived is what you
want.

Asking for a resolution is not the same as getting one. OpenCV's `set` returns
True whether or not the driver honoured the request, and V4L2 quietly
substitutes the nearest mode it does support. A camera that fell back to VGA
looks like a camera that is out of focus once its frames are scaled up for
display -- and since the ZED's lenses are fixed-focus, soft frames are nearly
always this rather than the optics. So the mode is set in the order the V4L2
backend wants (pixel format first, then size, then rate), then read back and
checked, and a mismatch raises instead of being left to show up as blur.

The camera will not say which camera it is over USB. Its serial number lives
in a camera register reached by a vendor-specific request, not in the USB
descriptor, so udev has nothing to name the device after and several identical
cameras on a bench look alike. The serial is what selects a calibration, so it
is read from the camera itself through the zed-open-capture shim -- see
`resolve_serial`. That means the shim is needed for a camera to find its own
calibration, though `serial=` still overrides it for a machine without one.

Two further traps are not this module's to fix, but are worth knowing about:

    USB 2     pins the camera to its VGA mode no matter what is requested, so
              a `CameraError` about resolution is worth answering by checking
              which port it is in.
    exposure  is not reachable from here. The ZED does not expose exposure or
              gain as standard V4L2 controls; `zed-open-capture` reaches them
              through the same kind of vendor-specific request as the serial,
              so manual exposure needs that library behind a small binding.
              Everything else here works without it.
"""

import dataclasses
import pathlib
import sys

import cv2
import numpy as np

from . import zed_controls
from .calibration import RESOLUTIONS, StereoCalibration

# What each mode runs at unless asked otherwise. These are the camera's maxima
# for the mode; the driver will accept lower rates.
DEFAULT_FPS = {"2K": 15, "FHD": 30, "HD": 60, "VGA": 100}

# Uncompressed 4:2:2 is the only thing a USB3 ZED emits -- there is no MJPEG
# mode to fall back to, which is also why a soft frame is never a compression
# artefact.
_FOURCC = "YUYV"

# Cameras are found through sysfs rather than by opening indices in turn: the
# card name says which nodes are ZEDs, and a UVC camera registers more than one
# node, of which only per-device index 0 carries video. The others return
# metadata, and opening one looks like a camera that produces nothing.
_SYSFS_V4L = pathlib.Path("/sys/class/video4linux")
_BY_PATH = pathlib.Path("/dev/v4l/by-path")
_CAPTURE_INDEX = "0"

# Asking a camera its serial means opening it, which only works while nothing
# is streaming it, so each answer is kept for the life of the process. Without
# this a second ZedCapture would have to interrupt the first to identify it.
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


class ZedCapture:
    """A ZED stereo camera, read through OpenCV.

    Opens the camera in one of its video modes and hands back the two eyes
    separately. Closes on the way out of a `with` block.

    Args:
        resolution: One of "2K", "FHD", "HD", "VGA". These are per-eye sizes;
            the frame off the wire is twice as wide.
        device: A `/dev/video*` path or an integer index. Found automatically
            when omitted, which is more reliable than guessing an index.
        serial: Which camera to open, and what its calibration is looked up
            by. Resolved from the machine when omitted -- see `resolve_serial`.
        fps: Frame rate. Defaults to the mode's maximum.
        exposure: Exposure to apply before streaming starts. Needs the
            zed-open-capture shim, since this is not a V4L2 control -- see
            `zed_controls`. Left as the camera had it when None.
        gain: Gain, on the same terms as exposure.
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
        self.serial = str(serial) if serial is not None else None
        self._calibration = None

        if device is None:
            found, resolved = find_device(serial)
            device, self.port = found.path, found.port
            self.serial = self.serial or resolved
        else:
            # An explicitly named node skips discovery, so ask the camera
            # directly -- while it is still free, before the stream below.
            self.port = ""
            if self.serial is None:
                try:
                    self.serial = zed_controls.read_serial(device)
                except zed_controls.ControlsError:
                    self.serial = None
        self.device = device
        requested_fps = DEFAULT_FPS[resolution] if fps is None else fps

        # Exposure and gain are not V4L2 controls on this camera, so they go
        # through zed-open-capture -- which opens the device, and so has to be
        # finished with it before the stream below starts. `zed_controls` has
        # the ordering and the one assumption it rests on.
        self.exposure = self.gain = None
        if exposure is not None or gain is not None:
            self.exposure, self.gain = zed_controls.apply(
                device,
                resolution=resolution,
                fps=requested_fps,
                exposure=exposure,
                gain=gain,
            )

        # Naming the backend matters: left to choose, OpenCV may pick
        # GStreamer, which handles these properties differently or ignores them.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2
        source = device if isinstance(device, int) else str(device)
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            raise CameraError(f"could not open {device}")

        # Order matters: the V4L2 backend applies the pixel format first, and a
        # size set before it can be reverted when the format changes.
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_FOURCC))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, requested_fps)

        # The shallowest queue the driver will give. A deep one means every
        # read hands back a frame the camera captured several frames ago, so a
        # loop that cannot quite keep up ends up looking steadily further into
        # the past rather than dropping frames and staying current. Not every
        # backend honours it, and a refusal is not worth reporting.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._check_mode(2 * width, height)

    def _check_mode(self, width, height):
        """Confirm the driver gave the mode that was asked for, not a near miss."""
        actual = (
            round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (width, height):
            self.close()
            raise CameraError(
                f"asked for {width}x{height} ({self.resolution}) but the driver "
                f"gave {actual[0]}x{actual[1]}. The usual causes are a USB 2 "
                "port, which pins the camera to VGA; a device node that is not "
                "the camera's capture node; or, on macOS, a backend that "
                "ignores resolution requests altogether. "
                "`v4l2-ctl --list-formats-ext` lists what the device will accept."
            )
        self.fps = self._capture.get(cv2.CAP_PROP_FPS)

    @property
    def pixel_format(self):
        """The negotiated pixel format, as its four-character code."""
        code = round(self._capture.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((code >> (8 * byte)) & 0xFF) for byte in range(4))

    @property
    def calibration(self):
        """This camera's factory calibration at the current resolution."""
        if self._calibration is None:
            self._calibration = _calibration_for(self.serial, self.resolution)
        return self._calibration

    def read_raw(self):
        """Grab one frame as it arrives: both eyes side by side in one image."""
        ok, frame = self._capture.read()
        if not ok:
            raise CameraError("frame grab failed -- the camera may have been unplugged")
        return frame

    def read(self):
        """Grab one frame and return it as `(left, right)`.

        The halves are copied rather than returned as views of the frame:
        splitting leaves them non-contiguous, and much of cv2 refuses an array
        it cannot address directly.
        """
        left, right = np.split(self.read_raw(), 2, axis=1)
        return np.ascontiguousarray(left), np.ascontiguousarray(right)

    def read_rectified(self):
        """Grab one frame and return the two eyes rectified.

        Needs the calibration, and remaps every pixel. For tracking a handful
        of points, grab with `read` and use `calibration.undistort_points`
        instead -- it is the same geometry for a fraction of the work.
        """
        return self.calibration.rectify_images(*self.read())

    def close(self):
        """Release the camera."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        serial = f"SN{self.serial}" if self.serial else "unknown serial"
        return f"<ZedCapture {serial} {self.resolution} {self.size[0]}x{self.size[1]}>"


class ZedOpenCapture:
    """A ZED read through zed-open-capture rather than OpenCV.

    Interchangeable with `ZedCapture` -- the same `read`, the same
    `calibration`, the same `with` block -- and the one to prefer. OpenCV's
    V4L2 path handed back torn frames from this camera, bands of misaligned
    YUYV, where ffmpeg reading the same device was clean; telling the kernel to
    discard the torn ones rather than pass them through then cost half the
    frame rate. This goes through the implementation Stereolabs wrote for the
    camera instead, so it needs the native shim built -- see native/README.md.

    One handle streams and controls at once, so `set_exposure` and `set_gain`
    take effect without stopping the stream, which `ZedCapture` cannot do.

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

"""ZED stereo camera over plain UVC, with no ZED SDK and no CUDA.

A USB3 ZED is a UVC-compliant device, so its frames can be pulled straight off
`/dev/video*` with OpenCV. That is the whole reason this module exists: the ZED
SDK requires an NVIDIA GPU and the lab machine has none, but nothing about
*capturing* needs one. `calibration.py` gets the calibration the same way, so
between them there is no part of the pipeline that wants the SDK.

Two things about this camera surprise people, and both are handled here.

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

Two further traps are not this module's to fix, but are worth knowing about:

    USB 2     pins the camera to its VGA mode no matter what is requested, so
              a `CameraError` about resolution is worth answering by checking
              which port it is in.
    exposure  is not reachable from here. The ZED does not expose exposure or
              gain as standard V4L2 controls; `zed-open-capture` reaches them
              through vendor-specific register writes, so manual exposure needs
              that library behind a small binding. Everything else here works
              without it.
"""

import pathlib
import re
import sys

import cv2
import numpy as np

from .calibration import RESOLUTIONS, CalibrationError, StereoCalibration

# What each mode runs at unless asked otherwise. These are the camera's maxima
# for the mode; the driver will accept lower rates.
DEFAULT_FPS = {"2K": 15, "FHD": 30, "HD": 60, "VGA": 100}

# Uncompressed 4:2:2 is the only thing a USB3 ZED emits -- there is no MJPEG
# mode to fall back to, which is also why a soft frame is never a compression
# artefact.
_FOURCC = "YUYV"

# udev names each node after the USB device here, so the camera can be found by
# what it is rather than by an index that moves when another webcam is plugged
# in. A UVC camera registers more than one node -- index0 is the one that
# carries video, the others carry metadata and return nothing useful.
_BY_ID = pathlib.Path("/dev/v4l/by-id")
_CAPTURE_NODE = "index0"
_NODE_SUFFIX = re.compile(r"-video-index\d+$")
_SERIAL_IN_NAME = re.compile(r"(\d{4,})")


class CameraError(RuntimeError):
    """The camera could not be opened, configured, or read from."""


def find_device(serial=None):
    """Locate a ZED's video node.

    Args:
        serial: If given, only match the camera with this serial number, which
            is what to pass when more than one camera is attached.

    Returns:
        `(path, serial)`, where serial is None if it could not be read out of
        the device name.
    """
    if not _BY_ID.is_dir():
        raise CameraError(
            f"{_BY_ID} does not exist, so there are no cameras to search. On "
            "Linux that usually means nothing is plugged in; elsewhere, pass "
            "device= explicitly."
        )

    found = []
    for entry in sorted(_BY_ID.iterdir()):
        if "ZED" not in entry.name.upper() or not entry.name.endswith(_CAPTURE_NODE):
            continue
        match = _SERIAL_IN_NAME.search(_NODE_SUFFIX.sub("", entry.name))
        found.append((entry.resolve(), match.group(1) if match else None))

    if not found:
        raise CameraError(f"no ZED capture node in {_BY_ID} -- is the camera plugged in?")

    if serial is None:
        if len(found) > 1:
            serials = ", ".join(str(s) for _, s in found)
            raise CameraError(f"more than one ZED attached ({serials}) -- pass serial=")
        return found[0]

    for path, device_serial in found:
        if device_serial == str(serial):
            return path, device_serial
    attached = ", ".join(str(s) for _, s in found)
    raise CameraError(f"no ZED with serial {serial} attached (found: {attached})")


class ZedCapture:
    """A ZED stereo camera, read through OpenCV.

    Opens the camera in one of its video modes and hands back the two eyes
    separately. Closes on the way out of a `with` block.

    Args:
        resolution: One of "2K", "FHD", "HD", "VGA". These are per-eye sizes;
            the frame off the wire is twice as wide.
        device: A `/dev/video*` path or an integer index. Found automatically
            when omitted, which is more reliable than guessing an index.
        serial: The camera's serial number. Read from the device name when
            omitted, and needed for `calibration` to know what to look up.
        fps: Frame rate. Defaults to the mode's maximum.
    """

    def __init__(self, resolution="HD", device=None, serial=None, fps=None):
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
            device, found_serial = find_device(serial)
            if self.serial is None:
                self.serial = found_serial
        self.device = device

        # Naming the backend matters: left to choose, OpenCV may pick GStreamer,
        # which handles these properties differently or ignores them.
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
        self._capture.set(cv2.CAP_PROP_FPS, DEFAULT_FPS[resolution] if fps is None else fps)
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
            if self.serial is None:
                raise CameraError(
                    "this camera's serial number is not known, so its "
                    "calibration cannot be looked up -- pass serial=, which is "
                    "printed on the label on the camera's back panel"
                )
            self._calibration = StereoCalibration.from_serial(self.serial, self.resolution)
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
        of points, grab with `read` and use
        `calibration.undistort_points` instead -- it is the same geometry for a
        fraction of the work.
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


if __name__ == "__main__":
    # A probe for the two failure modes above: it reports what the driver
    # actually negotiated, and writes the split eyes out so they can be looked
    # at 1:1 rather than scaled into a window.
    resolution = sys.argv[1] if len(sys.argv) > 1 else "HD"

    with ZedCapture(resolution) as camera:
        print(f"device    {camera.device}")
        print(f"serial    {camera.serial}")
        print(f"mode      {camera.resolution}, {camera.size[0]}x{camera.size[1]} per eye")
        print(f"format    {camera.pixel_format} at {camera.fps:g} fps")

        left, right = camera.read()
        cv2.imwrite("/tmp/zed_left.png", left)
        cv2.imwrite("/tmp/zed_right.png", right)
        print(f"wrote     /tmp/zed_left.png, /tmp/zed_right.png ({left.shape[1]}x{left.shape[0]})")
        print("          open them at 1:1 -- if they are sharp, the blur was display scaling")

        try:
            calibration = camera.calibration
        except (CameraError, CalibrationError) as error:
            print(f"calib     unavailable: {error}")
        else:
            print(f"focal     {calibration.focal_length:.1f} px")
            print(f"baseline  {1000 * calibration.baseline:.1f} mm")
            print(f"depth     {100 * calibration.depth_from_disparity(44.0):.1f} cm at 44 px disparity")

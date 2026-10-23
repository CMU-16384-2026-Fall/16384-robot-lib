"""Exposure, gain and the serial number, through zed-open-capture.

Three things on a ZED are reachable only through vendor-specific USB requests:
the exposure, the gain, and the serial number. OpenCV cannot make those
requests -- the camera does not publish exposure and gain as standard V4L2
controls, and does not publish the serial in its USB descriptor at all -- so
`camera.py` can do everything except these. `zed-open-capture` can make them,
and this is the ctypes binding onto a small C entry point over it.

Frames come from here too, and that was not the original plan. OpenCV's V4L2
path returned torn frames from this camera -- bands of misaligned YUYV -- where
ffmpeg reading the same device was clean, and once the kernel was told to drop
the torn ones instead of passing them through, half the frame rate went with
them. So capture moved to the implementation Stereolabs wrote for this camera.

One handle streams and controls at once, which is simpler than what came
before: exposure now takes effect without stopping the stream, and the question
of whether a setting survives a close and reopen does not arise.

The exposure a value means depends on the frame rate, because the library maps
it onto raw sensor registers whose range shrinks as the rate rises. So a
control session has to be opened at the same resolution and rate the camera
will then stream at, and `resolution` and `fps` are required here rather than
defaulted for that reason.

Nothing else in the library needs this module. Without the shim built, capture
and calibration work as before and only these three things are missing.
"""

import ctypes
import os
import pathlib
import re

import numpy as np

# The shim's resolution codes are zed-open-capture's RESOLUTION enum in
# declaration order, keyed by the names the rest of the library uses.
_RESOLUTION_CODE = {"2K": 0, "FHD": 1, "HD": 2, "VGA": 3}

# Matching its CAM_SENS_POS.
LEFT = 0
RIGHT = 1

# Long enough that a dropped frame does not look like a dead camera, short
# enough that a dead camera does not look like a slow one.
FRAME_TIMEOUT_MS = 200

LIBRARY_NAME = "libzed_oc_shim.so"
SHIM_ENV = "ZED_OC_SHIM"

_BUILD_HINT = (
    "Build it with `cd native && cmake -B build && cmake --build build`, "
    f"or set {SHIM_ENV} to its path. See native/README.md -- the rest of the "
    "library works without it."
)

_library = None


class ControlsError(RuntimeError):
    """The shim is missing, or the camera would not open to be configured."""


def _candidates():
    from_environment = os.environ.get(SHIM_ENV, "").strip()
    if from_environment:
        yield pathlib.Path(from_environment)
    package = pathlib.Path(__file__).resolve().parent
    # Where native/CMakeLists.txt leaves it in a source checkout, then beside
    # the package for an installed one, then wherever the loader can find it.
    yield package.parent.parent / "native" / "build" / LIBRARY_NAME
    yield package / LIBRARY_NAME
    yield pathlib.Path(LIBRARY_NAME)


def _signatures(library):
    handle = ctypes.c_void_p
    library.zed_oc_open.argtypes = [ctypes.c_int] * 4
    library.zed_oc_open.restype = handle
    library.zed_oc_close.argtypes = [handle]
    library.zed_oc_close.restype = None
    for name in ("zed_oc_serial", "zed_oc_get_auto"):
        getattr(library, name).argtypes = [handle]
        getattr(library, name).restype = ctypes.c_int
    for name in ("zed_oc_get_exposure", "zed_oc_get_gain", "zed_oc_set_auto",
                 "zed_oc_set_auto_white_balance", "zed_oc_set_white_balance"):
        getattr(library, name).argtypes = [handle, ctypes.c_int]
        getattr(library, name).restype = ctypes.c_int
    for name in ("zed_oc_set_exposure", "zed_oc_set_gain"):
        getattr(library, name).argtypes = [handle, ctypes.c_int, ctypes.c_int]
        getattr(library, name).restype = ctypes.c_int

    library.zed_oc_frame.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_ubyte),  # where to copy the frame
        ctypes.c_int,  # how many bytes that will hold
        ctypes.POINTER(ctypes.c_int),  # width, both eyes together
        ctypes.POINTER(ctypes.c_int),  # height
        ctypes.POINTER(ctypes.c_int),  # channels, 2 for YUV 4:2:2
        ctypes.POINTER(ctypes.c_ulonglong),  # frame id
        ctypes.POINTER(ctypes.c_ulonglong),  # timestamp, nanoseconds
        ctypes.c_int,  # timeout
    ]
    library.zed_oc_frame.restype = ctypes.c_int
    return library


def load_shim():
    """Load the shim library, or say how to build it. Cached after the first call."""
    global _library
    if _library is not None:
        return _library

    tried = []
    for candidate in _candidates():
        tried.append(str(candidate))
        try:
            _library = _signatures(ctypes.CDLL(str(candidate)))
        except OSError:
            continue
        return _library

    raise ControlsError(
        f"{LIBRARY_NAME} was not found (tried: {', '.join(tried)}). {_BUILD_HINT}"
    )


def available():
    """Whether the shim is built and loadable, for deciding without catching."""
    try:
        load_shim()
    except ControlsError:
        return False
    return True


def _device_id(device):
    """Turn a ZedDevice, a /dev/video* path or a number into the library's id."""
    if device is None:
        return -1  # the library picks the first camera it finds
    if isinstance(device, int):
        return device
    name = str(getattr(device, "node", device))
    trailing = re.search(r"(\d+)$", name)
    if trailing is None:
        raise ControlsError(f"cannot tell which video node {name!r} refers to")
    return int(trailing.group(1))


class ZedControls:
    """A control session on one camera, for as long as nothing else streams it.

    Opens the camera, so it must be closed before `ZedCapture` opens the same
    device -- use it as a context manager and the ordering looks after itself:

        with ZedControls(device, "HD", 60) as controls:
            controls.set_exposure(20)

    Args:
        device: A `ZedDevice`, a "videoN" name, or a node number. The library
            picks the first camera it finds when this is None.
        resolution: The mode the camera will be streamed at -- see the module
            docstring for why this has to match.
        fps: The frame rate it will be streamed at, for the same reason.
        verbose: zed-open-capture's own verbosity, 0 for silent.
    """

    def __init__(self, device=None, resolution="HD", fps=60, verbose=0):
        if resolution not in _RESOLUTION_CODE:
            raise ControlsError(
                f"unknown resolution {resolution!r}, expected one of "
                f"{', '.join(_RESOLUTION_CODE)}"
            )
        self._library = load_shim()
        self._buffer = None
        self._handle = self._library.zed_oc_open(
            _RESOLUTION_CODE[resolution], int(fps), _device_id(device), int(verbose)
        )
        if not self._handle:
            raise ControlsError(
                f"zed-open-capture could not open {device if device is not None else 'a camera'} "
                f"at {resolution}/{fps}. The usual cause is that something else "
                "has it open -- a ZedCapture in this process, or another "
                "program streaming it."
            )

    def _check(self):
        if self._handle is None:
            raise ControlsError("this control session is closed")

    @property
    def serial(self):
        """The serial number, read from the camera itself."""
        self._check()
        serial = self._library.zed_oc_serial(self._handle)
        if serial <= 0:
            raise ControlsError(f"the camera returned no serial number ({serial})")
        return str(serial)

    def frame(self, timeout_ms=FRAME_TIMEOUT_MS):
        """The newest frame, as `(array, frame_id, timestamp_ns)`.

        The array is raw YUV 4:2:2 shaped `(height, width, channels)` with both
        eyes side by side; `camera.ZedOpenCapture` is what turns it into two
        BGR images. It is a view of a buffer this session reuses, so the next
        call overwrites it -- copy or convert before asking for another.

        `frame_id` increases once per frame the camera produced, so the same
        value twice means this was asked faster than the camera delivers rather
        than that the scene stood still.
        """
        self._check()
        width, height, channels = (ctypes.c_int() for _ in range(3))
        frame_id, timestamp = ctypes.c_ulonglong(), ctypes.c_ulonglong()

        for _ in range(2):
            capacity = 0 if self._buffer is None else self._buffer.size
            pointer = (
                None
                if self._buffer is None
                else self._buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            )
            status = self._library.zed_oc_frame(
                self._handle,
                pointer,
                capacity,
                ctypes.byref(width),
                ctypes.byref(height),
                ctypes.byref(channels),
                ctypes.byref(frame_id),
                ctypes.byref(timestamp),
                int(timeout_ms),
            )
            if status == 0:
                break
            if status == -3:
                # The first call has no buffer yet, and the shim answers with
                # the size it wanted; allocate that and ask again.
                self._buffer = np.empty(
                    width.value * height.value * channels.value, dtype=np.uint8
                )
                continue
            if status == -2:
                raise ControlsError(
                    f"no frame within {timeout_ms} ms -- the camera may have "
                    "stopped streaming"
                )
            raise ControlsError(f"frame grab failed with status {status}")
        else:
            raise ControlsError("the camera kept asking for a larger buffer")

        shape = (height.value, width.value, channels.value)
        size = shape[0] * shape[1] * shape[2]
        return self._buffer[:size].reshape(shape), frame_id.value, timestamp.value

    def exposure(self, side=LEFT):
        """The current exposure of one sensor."""
        self._check()
        return self._library.zed_oc_get_exposure(self._handle, side)

    def set_exposure(self, value, side=None):
        """Set exposure, on both sensors unless one is named.

        Switches off automatic exposure and gain, which the library does as a
        side effect of this call. Read it back to see what was accepted: the
        value maps onto sensor registers whose range depends on the frame rate,
        so it is not necessarily the number handed in.
        """
        self._check()
        for sensor in (LEFT, RIGHT) if side is None else (side,):
            self._library.zed_oc_set_exposure(self._handle, sensor, int(value))

    def gain(self, side=LEFT):
        """The current gain of one sensor."""
        self._check()
        return self._library.zed_oc_get_gain(self._handle, side)

    def set_gain(self, value, side=None):
        """Set gain, on both sensors unless one is named."""
        self._check()
        for sensor in (LEFT, RIGHT) if side is None else (side,):
            self._library.zed_oc_set_gain(self._handle, sensor, int(value))

    @property
    def auto_exposure(self):
        """Whether the camera is running automatic exposure and gain."""
        self._check()
        return bool(self._library.zed_oc_get_auto(self._handle))

    @auto_exposure.setter
    def auto_exposure(self, active):
        self._check()
        self._library.zed_oc_set_auto(self._handle, 1 if active else 0)

    def set_white_balance(self, value=None):
        """Set white balance, or hand it back to the camera when given None."""
        self._check()
        if value is None:
            self._library.zed_oc_set_auto_white_balance(self._handle, 1)
        else:
            self._library.zed_oc_set_white_balance(self._handle, int(value))

    def close(self):
        """Release the camera, which has to happen before anything streams it."""
        if self._handle is not None:
            self._library.zed_oc_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_serial(device=None):
    """Ask a camera its serial number.

    Opens it briefly, so nothing else may be streaming it at the time. VGA is
    used because the mode does not matter for this and it is the cheapest to
    start.
    """
    with ZedControls(device, resolution="VGA", fps=100) as controls:
        return controls.serial


def apply(device=None, resolution="HD", fps=60, exposure=None, gain=None):
    """Set exposure and/or gain on a camera, then let go of it.

    What `ZedCapture` calls before opening its stream. Returns what the camera
    reports afterwards as `(exposure, gain)`, which is not always what was
    asked for.
    """
    with ZedControls(device, resolution=resolution, fps=fps) as controls:
        if exposure is not None:
            controls.set_exposure(exposure)
        if gain is not None:
            controls.set_gain(gain)
        return controls.exposure(), controls.gain()

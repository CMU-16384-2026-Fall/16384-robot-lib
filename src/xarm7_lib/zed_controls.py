"""Frames, exposure, gain, colour and the serial number, through zed-open-capture.

The exposure, the gain and the serial number are reachable only through
vendor-specific USB requests: the camera does not publish exposure and gain as
standard V4L2 controls, and does not publish the serial in its USB descriptor
at all. Frames come from here too, so one handle streams and controls at once
and exposure takes effect without stopping the stream.

This is the ctypes binding onto a small C entry point over the library; see
`native/`.

The exposure a value means depends on the frame rate, because the library maps
it onto raw sensor registers whose range shrinks as the rate rises. So a
control session has to be opened at the same resolution and rate the camera
will then stream at, and `resolution` and `fps` are required here rather than
defaulted for that reason.

Colour is the other half of this. The camera's ISP runs automatic white
balance by default, so the same scene comes out a different colour on
different days -- and the two eyes do not track each other exactly while it
is on, which puts a bias between them that no amount of care downstream can
undo. `CONTROLS` covers the settings that decide this, `profile` reads them
all back, and `apply` puts a saved one on again, so a session can be
reproduced rather than remembered.

Without the shim built there is no capture and no calibration, since the
serial number is what selects one.
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

# The whole-camera ISP controls, as `name: (code, low, high)`. The codes are
# the shim's `Control` enum and have to stay in step with it; the ranges are
# the ones zed-open-capture documents on its setters, and the camera clamps to
# them anyway -- they are here to fail loudly on an obvious mistake rather
# than to silently accept it.
#
# The last two are flags rather than levels, but they are read and written
# exactly like the rest, so they live here too: `auto_white_balance` hands
# colour back to the camera and `aec_agc` hands it exposure and gain, which
# between them are what make one day's footage differ from another's.
CONTROLS = {
    "brightness": (0, 0, 8),
    "contrast": (1, 0, 8),
    "hue": (2, 0, 11),
    "saturation": (3, 0, 8),
    "sharpness": (4, 0, 8),
    "gamma": (5, 1, 9),
    "white_balance": (6, 2800, 6500),
    "auto_white_balance": (7, 0, 1),
    "aec_agc": (8, 0, 1),
}

# The two that hand a setting back to the camera. They go on last in `apply`,
# because writing a value switches the matching one off as a side effect.
AUTO_CONTROLS = ("auto_white_balance", "aec_agc")

# Everything `profile` records. Exposure and gain are not in CONTROLS because
# they are per-sensor, but they belong in a profile: a fixed colour with a
# drifting exposure is still not reproducible.
PROFILE_KEYS = ("exposure", "gain") + tuple(CONTROLS)

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
    library.zed_oc_serial.argtypes = [handle]
    library.zed_oc_serial.restype = ctypes.c_int
    for name in ("zed_oc_get_exposure", "zed_oc_get_gain",
                 "zed_oc_get_control", "zed_oc_reset_control"):
        getattr(library, name).argtypes = [handle, ctypes.c_int]
        getattr(library, name).restype = ctypes.c_int
    for name in ("zed_oc_set_exposure", "zed_oc_set_gain", "zed_oc_set_control"):
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


def _control_range(name):
    try:
        return CONTROLS[name]
    except KeyError:
        raise ControlsError(
            f"unknown control {name!r}, expected one of {', '.join(CONTROLS)}"
        ) from None


def _control_code(name):
    return _control_range(name)[0]


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
    """A session on one camera: its frames, its exposure, its gain, its serial.

    Holds the camera open, so only one of these can exist per camera at a
    time. `camera.ZedOpenCapture` wraps one and is what most code should use;
    this is the layer underneath.

        with ZedControls(device, "HD", 60) as session:
            session.set_exposure(20)
            frame, frame_id, timestamp = session.frame()

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
                f"at {resolution}/{fps}. The usual cause is that something "
                "else has it open -- another session in this process, or "
                "another program streaming it."
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

    def set_white_balance(self, value=None):
        """Fix white balance in kelvin, or hand it back to the camera for None.

        A convenience over the table for the one pairing that is always
        wanted together: a value only means anything once the camera has
        stopped choosing its own.
        """
        if value is None:
            return self.set_control("auto_white_balance", 1)
        return self.set_control("white_balance", value)

    def control(self, name):
        """One ISP control by name; see `CONTROLS` for what there is."""
        self._check()
        code = _control_code(name)
        value = self._library.zed_oc_get_control(self._handle, code)
        if value == -2:
            raise ControlsError(f"the shim does not know the control {name!r}")
        return value

    def set_control(self, name, value):
        """Set one ISP control by name, and return what the camera then reports.

        The value comes back read rather than assumed, because the camera
        clamps and quantises -- white balance in particular moves in steps of
        a hundred kelvin or so, and a profile written from these readings is
        one that will reapply exactly.
        """
        self._check()
        code, low, high = _control_range(name)
        value = int(value)
        if not low <= value <= high:
            raise ControlsError(
                f"{name} must be between {low} and {high}, not {value}"
            )
        if self._library.zed_oc_set_control(self._handle, code, value) == -2:
            raise ControlsError(f"the shim does not know the control {name!r}")
        return self.control(name)

    def reset_control(self, name):
        """Put one ISP control back to the library's default."""
        self._check()
        code = _control_code(name)
        if self._library.zed_oc_reset_control(self._handle, code) == -2:
            raise ControlsError(f"the shim does not know the control {name!r}")
        return self.control(name)

    def profile(self):
        """Everything that decides what a frame looks like, as a plain dict.

        Write it out at the end of a session and `apply` it at the start of
        the next one and the two are comparable. Without that the camera
        starts on automatic white balance and the colour depends on whatever
        it happened to settle on, which is not a property of the scene.
        """
        self._check()
        settings = {"exposure": self.exposure(), "gain": self.gain()}
        settings.update({name: self.control(name) for name in CONTROLS})
        return settings

    def apply(self, settings, strict=True):
        """Put a profile back on, and return what the camera actually took.

        The manual values go on first and the two automatic modes last, which
        is the only order that round-trips. Writing an exposure or a white
        balance switches the matching automatic mode off as a side effect --
        that is the camera's behaviour, not this wrapper's -- so setting the
        modes first would leave them off whatever the profile asked for, and
        a profile saved with automatic white balance on would come back with
        it off. Doing it this way, a profile that wants automatic gets it, and
        one that wants fixed values has already switched automatic off by
        setting them, so the last pass only confirms what is already true.
        """
        self._check()
        unknown = set(settings) - set(PROFILE_KEYS)
        if unknown and strict:
            raise ControlsError(
                f"not camera settings: {', '.join(sorted(unknown))}. "
                f"Known ones are {', '.join(PROFILE_KEYS)}"
            )

        if "exposure" in settings:
            self.set_exposure(settings["exposure"])
        if "gain" in settings:
            self.set_gain(settings["gain"])
        for name in CONTROLS:
            if name not in AUTO_CONTROLS and name in settings:
                self.set_control(name, settings[name])
        for name in AUTO_CONTROLS:
            if name in settings:
                self.set_control(name, settings[name])
        return self.profile()

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

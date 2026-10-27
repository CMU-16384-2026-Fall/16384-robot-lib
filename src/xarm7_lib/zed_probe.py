"""Check that a ZED is being captured properly.

    python -m xarm7_lib.zed_probe                   # what is attached, and how
    python -m xarm7_lib.zed_probe --resolution VGA
    python -m xarm7_lib.zed_probe --exposure 10     # needs the native shim

The probe exists because the two things that go wrong with this camera are both
silent. A mode the driver quietly substituted shows up as frames that look out
of focus, and a camera that cannot be asked its serial shows up as a missing
calibration -- so this prints what was actually negotiated, and writes the two
eyes out to be looked at at 1:1 rather than scaled into a window.

It is a module of its own rather than an `if __name__ == "__main__"` block in
`camera.py` because the package imports `camera`, and `python -m` on a module
that is already in `sys.modules` executes a second copy of it. Python warns
about that, and it would leave the probe and the library disagreeing about what
`CameraError` is.
"""

import argparse
import pathlib
import sys

import cv2

from .calibration import CalibrationError
from .camera import CameraError, ZedOpenCapture, devices, resolve_serial
from .zed_controls import ControlsError, available


def _device_argument(value):
    """A `/dev/video*` path, or a bare index like the one cv2 would take."""
    return int(value) if value.isdigit() else value


def _report_devices():
    attached = devices()
    print(f"{len(attached)} ZED capture node(s) attached:")
    for device in attached:
        serial = resolve_serial(device)
        print(f"  {device.path}")
        print(f"    port        {device.port or '(no by-path link)'}")
        print(f"    usb serial  {device.usb_serial or '(not published, as expected)'}")
        print(f"    serial      {serial or 'unknown'}")
    if not any(resolve_serial(device) for device in attached):
        print(
            "\n  No serial read back. The camera only gives it up through the\n"
            "  zed-open-capture shim, so either that is not built or something\n"
            "  else is streaming the camera. --serial skips the question."
        )
    print(f"\nnative shim: {'available' if available() else 'not built (see native/README.md)'}")
    print()
    return attached


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolution", default="HD", help="2K, FHD, HD or VGA (default HD)")
    parser.add_argument("--serial", help="which camera to open, if several are attached")
    parser.add_argument(
        "--device",
        type=_device_argument,
        help="open this node (a /dev/video* path or an index) instead of "
        "searching for one, for when enumeration is not available",
    )
    parser.add_argument("--exposure", type=int, help="exposure to apply (needs the shim)")
    parser.add_argument("--gain", type=int, help="gain to apply (needs the shim)")
    parser.add_argument(
        "--out", default="/tmp", help="directory for the two eye images (default /tmp)"
    )
    args = parser.parse_args(argv)

    if args.device is None:
        _report_devices()
    else:
        # Enumeration is what fails on a machine with no v4l2, so naming the
        # node has to skip it rather than report it first.
        print(f"opening {args.device} as given, without searching\n")

    with ZedOpenCapture(
        args.resolution,
        device=args.device,
        serial=args.serial,
        exposure=args.exposure,
        gain=args.gain,
    ) as camera:
        print(f"opened    {camera.device}")
        if camera.exposure is not None:
            print(f"exposure  {camera.exposure} (gain {camera.gain}) as the camera reports it")
        print(f"serial    {camera.serial or 'unknown'}")
        print(f"mode      {camera.resolution}, {camera.size[0]}x{camera.size[1]} per eye")
        print(f"format    {camera.pixel_format} at {camera.fps:g} fps")

        # Worth printing even though nothing here changes it: automatic white
        # balance is the reason two days' footage can differ with everything
        # else held the same, and it is invisible in the images themselves.
        try:
            settings = camera.profile()
        except ControlsError as error:
            print(f"colour    unavailable: {error}")
        else:
            print("colour    " + "  ".join(
                f"{name} {settings[name]}" for name in sorted(settings)
            ))
            auto = [name for name in ("aec_agc", "auto_white_balance")
                    if settings.get(name)]
            if auto:
                print(f"          {' and '.join(auto)} ON -- the colour will "
                      "drift between runs; pin it with a saved profile")

        left, right = camera.read()
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, image in (("left", left), ("right", right)):
            cv2.imwrite(str(out / f"zed_{name}.png"), image)
        print(f"wrote     {out / 'zed_left.png'} and {out / 'zed_right.png'}")
        print(f"          {left.shape[1]}x{left.shape[0]} each -- open them at 1:1; if they")
        print("          are sharp there, the blur was display scaling")

        try:
            calibration = camera.calibration
        except (CameraError, CalibrationError) as error:
            print(f"calib     unavailable: {error}")
            return 1

        focal, baseline = calibration.focal_length, calibration.baseline
        print(f"calib     {calibration!r}")
        print(f"focal     {focal:.1f} px")
        print(f"baseline  {1000 * baseline:.1f} mm")
        # What a pixel of disparity error is worth, which is the number that
        # decides whether this camera can resolve what you are tracking.
        for depth in (0.5, 1.0, 2.0):
            disparity = focal * baseline / depth
            print(
                f"          at {depth:.1f} m: {disparity:5.1f} px disparity, "
                f"{100 * depth**2 / (focal * baseline):.2f} cm of depth per px"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CameraError, CalibrationError, ControlsError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

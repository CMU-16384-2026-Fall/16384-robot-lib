# zed_oc_shim

A C entry point onto [zed-open-capture](https://github.com/stereolabs/zed-open-capture)
for the three things OpenCV cannot reach on a ZED: the **serial number**, the
**exposure** and the **gain**. They go through vendor-specific USB requests
rather than standard V4L2 controls, so no amount of `cv2.CAP_PROP_*` will get
at them.

Everything else in `xarm7_lib` works without this. Capture and calibration need
only OpenCV; without the shim you lose exposure control and have to register
each camera's serial by hand.

## Building

Install `zed-open-capture` first. The IMU is not used here, so its sensor
module — and with it the `hidapi` dependency and the udev rule — can be left
out:

```bash
git clone https://github.com/stereolabs/zed-open-capture
cd zed-open-capture && mkdir build && cd build
cmake .. -DBUILD_SENSORS=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install && sudo ldconfig
```

Prerequisites for it: `build-essential cmake libusb-1.0-0-dev`.

Then build the shim:

```bash
cd native && cmake -B build && cmake --build build
```

That leaves `native/build/libzed_oc_shim.so`, which `zed_controls.py` looks for
without being told. Set `ZED_OC_SHIM` to its path to put it somewhere else.

## Checking it

With a camera attached and nothing else streaming it:

```bash
python -m xarm7_lib.zed_probe --register auto
```

which asks the camera its serial number and records it. Then

```bash
python -m xarm7_lib.zed_probe --exposure 20
```

## The one thing hardware has to confirm

The shim opens the camera, applies settings and closes it again, leaving the
stream to OpenCV, because only one owner can stream a V4L2 device at a time.
That assumes the camera keeps its exposure and gain registers across a close
and reopen. If it does not, the settings will appear to take and then be gone
by the time frames arrive — `--exposure 1` should give an obviously dark image,
so the test is quick. The fallback is to let zed-open-capture own the capture
as well, which is a larger change than this directory.

# zed_oc_shim

A C entry point onto [zed-open-capture](https://github.com/stereolabs/zed-open-capture)
for everything OpenCV cannot do with a ZED: the **serial number**, the
**exposure**, the **gain**, and — as it turned out — the **frames**.

The first three go through vendor-specific USB requests rather than standard
V4L2 controls, so no amount of `cv2.CAP_PROP_*` reaches them. Frames are here
for a different reason: OpenCV's V4L2 path returned torn frames from this
camera, bands of misaligned YUYV, where `ffmpeg` reading the same `/dev/video*`
was clean. Telling the kernel to discard the torn frames instead of passing
them through (`uvcvideo nodrop=0`) then cost half the frame rate, which showed
how often it was happening. So capture moved to the implementation Stereolabs
wrote for this camera.

`ZedOpenCapture` is the result and is the one to use; `ZedCapture`, the OpenCV
one, is kept for comparison. Without the shim built you get neither frames nor
calibration, since the serial number is what selects a calibration.

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
python -m xarm7_lib.zed_probe
```

The serial it prints is read from the camera itself, which only works with the
shim built -- that line alone confirms it is linked and working. Then

```bash
python -m xarm7_lib.zed_probe --exposure 20
```

## One handle does both

Because the same handle streams and controls, exposure and gain change without
stopping the stream — `ZedOpenCapture.set_exposure` takes effect on the next
frame. That also retires the question the earlier design turned on, of whether
a setting survives the camera being closed and reopened: nothing is reopened.

## If the frame rate is low

`uvcvideo`'s `nodrop` parameter decides whether incomplete frames reach
userspace. With `nodrop=1` they do, and OpenCV rendered them as bands; with
`nodrop=0` they are discarded, and the frame rate falls by however often they
occur. zed-open-capture may well cope with partial frames itself — which is the
likely reason something set `nodrop=1` on this machine in the first place — so
if the native path also runs slow, `nodrop=1` is worth retrying **with this
backend**, where the banding it caused under OpenCV may not appear.

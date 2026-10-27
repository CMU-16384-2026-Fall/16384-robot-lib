# zed_oc_shim

A C entry point onto [zed-open-capture](https://github.com/stereolabs/zed-open-capture)
for a ZED's **frames**, **serial number**, **exposure**, **gain** and
**colour** — white balance, brightness, contrast, hue, saturation, sharpness
and gamma.

Serial, exposure and gain go through vendor-specific USB requests rather than
standard V4L2 controls. Frames come from the same handle, so one session
streams and controls at once.

Exposure and gain have their own entry points because they are per-sensor and
need an argument the table has no room for. Everything else that decides how a
frame looks — white balance, automatic white balance, automatic exposure and
gain (`aec_agc`), brightness, contrast, hue, saturation, sharpness, gamma — is
one setting for the whole camera and shares a numbered `zed_oc_get_control` /
`zed_oc_set_control` / `zed_oc_reset_control` trio, so adding one is a line in
the shim's `Control` enum and a line in `CONTROLS` in `zed_controls.py` rather
than three new exported symbols. The two tables have to stay in step; nothing
checks that for you.

## Keeping the colour the same between sessions

The camera powers on with automatic white balance, so the same scene comes out
a different colour on different days — and the two eyes do not track each
other exactly while it is on, which biases anything that thresholds on colour
in one eye against the other. Save a profile once and hand it back:

```bash
python light_stick.py --save-colour lab.json      # whatever it is set to now
python light_stick.py --colour lab.json           # ... and again tomorrow
```

`python -m xarm7_lib.zed_probe` prints the current settings and warns when
either automatic mode is on. In code it is `camera.profile()` and
`ZedOpenCapture(..., controls=saved)`.

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

Then build the shim — and rebuild it after pulling a change to
`zed_oc_shim.cpp`, since `zed_controls.py` binds its symbols by name and an
older `.so` will not have them all:

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

## If the frame rate is low

`uvcvideo`'s `nodrop` parameter decides whether incomplete frames reach
userspace. With `nodrop=0` they are discarded and the frame rate falls by
however often they occur; with `nodrop=1` they are passed through, and whether
that is better depends on how well partial frames are handled downstream. Both
are worth trying. `light_stick.py --survey` measures the result per mode.

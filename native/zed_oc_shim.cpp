// A C entry point onto the three things zed-open-capture can do that OpenCV
// cannot: read the camera's serial number, and set its exposure and gain.
// Those go through vendor-specific USB requests rather than standard V4L2
// controls, which is why OpenCV cannot reach them and this file exists.
//
// Capturing frames is deliberately not here. A V4L2 device can only be
// streamed by one owner at a time, and camera.py already streams it through
// OpenCV, so this library is opened around that rather than replacing it --
// see zed_controls.py for the ordering that requires.

#include <videocapture.hpp>

using sl_oc::video::CAM_SENS_POS;
using sl_oc::video::FPS;
using sl_oc::video::RESOLUTION;
using sl_oc::video::VideoCapture;
using sl_oc::video::VideoParams;

namespace {

VideoCapture *as_capture(void *handle) {
    return static_cast<VideoCapture *>(handle);
}

CAM_SENS_POS side_of(int side) {
    // The Python side passes 0 for left and 1 for right, matching the enum.
    return side == 1 ? CAM_SENS_POS::RIGHT : CAM_SENS_POS::LEFT;
}

RESOLUTION resolution_of(int code) {
    switch (code) {
        case 0: return RESOLUTION::HD2K;
        case 1: return RESOLUTION::HD1080;
        case 2: return RESOLUTION::HD720;
        default: return RESOLUTION::VGA;
    }
}

FPS fps_of(int rate) {
    switch (rate) {
        case 15: return FPS::FPS_15;
        case 30: return FPS::FPS_30;
        case 60: return FPS::FPS_60;
        default: return FPS::FPS_100;
    }
}

}  // namespace

extern "C" {

// Returns an opaque handle, or null if the camera could not be opened --
// which is what happens when something else is already streaming it.
void *zed_oc_open(int resolution, int fps, int device_id, int verbose) {
    VideoParams params;
    params.res = resolution_of(resolution);
    params.fps = fps_of(fps);
    params.verbose = verbose;

    VideoCapture *capture = new VideoCapture(params);
    if (!capture->initializeVideo(device_id)) {
        delete capture;
        return nullptr;
    }
    return capture;
}

void zed_oc_close(void *handle) {
    delete as_capture(handle);
}

int zed_oc_serial(void *handle) {
    if (handle == nullptr) return -1;
    return as_capture(handle)->getSerialNumber();
}

int zed_oc_get_exposure(void *handle, int side) {
    if (handle == nullptr) return -1;
    return as_capture(handle)->getExposure(side_of(side));
}

// The library's setters return void, so these report only that the call was
// made. Read the value back to find out what the camera accepted.
int zed_oc_set_exposure(void *handle, int side, int value) {
    if (handle == nullptr) return -1;
    as_capture(handle)->setExposure(side_of(side), value);
    return 0;
}

int zed_oc_get_gain(void *handle, int side) {
    if (handle == nullptr) return -1;
    return as_capture(handle)->getGain(side_of(side));
}

int zed_oc_set_gain(void *handle, int side, int value) {
    if (handle == nullptr) return -1;
    as_capture(handle)->setGain(side_of(side), value);
    return 0;
}

int zed_oc_get_auto(void *handle) {
    if (handle == nullptr) return -1;
    return as_capture(handle)->getAECAGC() ? 1 : 0;
}

int zed_oc_set_auto(void *handle, int active) {
    if (handle == nullptr) return -1;
    return as_capture(handle)->setAECAGC(active != 0);
}

int zed_oc_set_auto_white_balance(void *handle, int active) {
    if (handle == nullptr) return -1;
    as_capture(handle)->setAutoWhiteBalance(active != 0);
    return 0;
}

int zed_oc_set_white_balance(void *handle, int value) {
    if (handle == nullptr) return -1;
    as_capture(handle)->setWhiteBalance(value);
    return 0;
}

}  // extern "C"

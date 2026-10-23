// A C entry point onto zed-open-capture: the camera's serial number, its
// exposure and gain, and its frames.
//
// The first three go through vendor-specific USB requests rather than standard
// V4L2 controls, which is why OpenCV cannot reach them. Frames are here for a
// different reason: OpenCV's V4L2 path was handing back torn frames from this
// camera where ffmpeg reading the same device was clean, so capture moved to
// the implementation Stereolabs wrote for it. One handle streams and controls
// at once, which also means exposure can change without stopping the stream.

#include <videocapture.hpp>

#include <cstring>

using sl_oc::video::CAM_SENS_POS;
using sl_oc::video::Frame;
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

// Copies the newest frame out as raw YUV 4:2:2, both eyes side by side.
// Returns 0 on success, -1 for a dead handle, -2 when no frame arrived inside
// the timeout, and -3 when `out` is too small -- in which case the size
// fields are still filled in, so a caller can size its buffer and retry.
//
// The copy is immediate and deliberate: getLastFrame hands back a reference
// to the capture thread's own frame, which it is free to overwrite.
int zed_oc_frame(void *handle, unsigned char *out, int capacity, int *width,
                 int *height, int *channels, unsigned long long *frame_id,
                 unsigned long long *timestamp, int timeout_ms) {
    if (handle == nullptr) return -1;

    const Frame &frame = as_capture(handle)->getLastFrame(timeout_ms);
    if (frame.data == nullptr || frame.width == 0 || frame.height == 0) return -2;

    if (width != nullptr) *width = frame.width;
    if (height != nullptr) *height = frame.height;
    if (channels != nullptr) *channels = frame.channels;
    // frame_id lets the caller tell a genuinely new frame from the same one
    // handed back twice, which is what "last frame" means when it is asked
    // for faster than the camera produces.
    if (frame_id != nullptr) *frame_id = frame.frame_id;
    if (timestamp != nullptr) *timestamp = frame.timestamp;

    const int needed =
        static_cast<int>(frame.width) * static_cast<int>(frame.height) *
        static_cast<int>(frame.channels);
    if (out == nullptr || capacity < needed) return -3;

    std::memcpy(out, frame.data, static_cast<size_t>(needed));
    return 0;
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

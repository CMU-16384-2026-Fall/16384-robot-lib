"""Finding the table tag in the camera: the OpenCV side of recovering a knocked camera.

`handeye.py` has the reasoning and the arithmetic, and imports this only when it
is about to open a camera, so reading a calibration still needs no OpenCV.

The tag is read at 2K whatever the calibration was made at. A 45 mm tag at
1.5 m, foreshortened by lying flat on the table, is as little as 11 px tall in
HD: it decodes intermittently, and its corners wander by a large fraction of a
pixel as the view changes. Rendered through this detector and corrected for a
bump, the error half a metre from the tag is 5-15 mm in HD and about 1 mm at 2K.
An 80 mm tag in HD managed about 1 mm at 1.25 m and went back to 10-20 mm at
1.5 m, so resolution is what matters. Real frames will be worse than rendered ones.
"""

import math
import time

import cv2
import numpy as np

from .camera import ZedOpenCapture
from .handeye import HandEyeError, Intrinsics, TagReference

# The last id in the dictionary the ChArUco board uses, because the board's own
# markers count up from 0 and may be in view at the same time.
TAG_ID = 49
TAG_DICTIONARY = "DICT_4X4_50"
TAG_LENGTH = 0.045  # m, the side of the black square
TAG_RESOLUTION = "2K"

LOOK_FRAMES = 15  # about a second at 2K
RECORD_FRAMES = 45  # the reference is compared against for months, so it gets more
SETTLE_FRAMES = 5  # discarded first, while auto-exposure finds its feet
FRAME_QUORUM = 0.6  # of those asked for, or the tag is not reliably in view
ATTEMPTS_PER_FRAME = 2  # how many frames may be looked at for each one wanted

# Rectified eyes agree on the row, so corners that do not are not the same thing
# seen twice.
MAX_ROW_ERROR = 4.0  # px


def aruco_dictionary(name):
    """One of OpenCV's predefined dictionaries, by name."""
    code = getattr(cv2.aruco, name, None)
    if not isinstance(code, int):
        available = sorted(
            attribute
            for attribute in dir(cv2.aruco)
            if attribute.startswith("DICT_")
        )
        raise HandEyeError(
            f"unknown ArUco dictionary {name!r}. Available: {', '.join(available)}"
        )
    return cv2.aruco.getPredefinedDictionary(code)


def tag_detector(dictionary=TAG_DICTIONARY):
    """A marker detector set up for a tag only tens of pixels across.

    OpenCV does no corner refinement unless asked, and sub-pixel corners are the
    whole measurement here. The small window keeps it from reaching into the
    tag's own bits, and the perimeter limit is lowered because the default drops
    a tag this small at some resolutions.
    """
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 3
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.minMarkerPerimeterRate = 0.01
    return cv2.aruco.ArucoDetector(aruco_dictionary(dictionary), parameters)


def find_tag(detector, image, tag_id):
    """One tag's corners in one image, `(4, 2)` in TL TR BR BL order, or None."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None
    for found, marker in zip(corners, np.asarray(ids).ravel()):
        if int(marker) == tag_id:
            return np.asarray(found, dtype=float).reshape(4, 2)
    return None


def capture_tag(camera, tag_id, dictionary=TAG_DICTIONARY, frames=LOOK_FRAMES):
    """The tag's corners over several frames. `(left, right, noise_px)` or None.

    The median rather than the mean, because a corner occasionally jumps a pixel
    or more on a tag this small, and one such frame should not move the answer.
    `noise_px` is how far that median can be trusted, read off the scatter.
    None means it was not seen in enough frames to believe, and that comes as
    soon as it cannot be: an arm parked in front of the tag is an ordinary thing
    for a script to start with, and should not cost it a long wait.
    """
    detector = tag_detector(dictionary)
    for _ in range(SETTLE_FRAMES):
        camera.read_rectified()

    lefts, rights = [], []
    attempts = ATTEMPTS_PER_FRAME * frames
    needed = math.ceil(FRAME_QUORUM * frames)
    for attempt in range(attempts):
        if len(lefts) >= frames or len(lefts) + attempts - attempt < needed:
            break
        left_image, right_image = camera.read_rectified()
        left = find_tag(detector, left_image, tag_id)
        if left is None:
            continue
        right = find_tag(detector, right_image, tag_id)
        if right is None:
            continue
        if np.max(np.abs(left[:, 1] - right[:, 1])) > MAX_ROW_ERROR:
            continue
        if np.any(left[:, 0] - right[:, 0] <= 0):
            continue
        lefts.append(left)
        rights.append(right)

    if len(lefts) < needed:
        return None
    left, right = np.median(lefts, axis=0), np.median(rights, axis=0)
    # MAD to a standard deviation, then the standard error of a median.
    scatter = 1.4826 * np.median(np.abs(np.asarray(lefts) - left))
    return left, right, float(1.25 * scatter / np.sqrt(len(lefts)))


def look_for_tag(reference, serial=None):
    """Open the camera and find a recorded tag. As `capture_tag` reports it."""
    with ZedOpenCapture(resolution=reference.resolution, serial=serial) as camera:
        return capture_tag(camera, reference.tag_id, reference.dictionary)


def record_reference(camera, tag_id=TAG_ID, dictionary=TAG_DICTIONARY,
                     length=TAG_LENGTH):
    """Record what the tag looks like now. A `TagReference`, or None if not seen.

    Whatever calibration this gets filed beside is being vouched for: from here
    on, this is what "the camera has not moved" means.
    """
    sighting = capture_tag(camera, tag_id, dictionary, frames=RECORD_FRAMES)
    if sighting is None:
        return None
    return TagReference(
        tag_id=int(tag_id),
        dictionary=dictionary,
        length=float(length),
        resolution=camera.resolution,
        intrinsics=Intrinsics.from_stereo(camera.calibration),
        left=sighting[0].tolist(),
        right=sighting[1].tolist(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

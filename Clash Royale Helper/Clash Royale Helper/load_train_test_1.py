import os
import numpy as np

# Setting up data
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths

BASE_DIR = os.path.dirname(__file__)


def loadTrainingImages1():
    """Load card training images from ``trainData`` folder."""
    imagePaths = sorted(list(paths.list_images(os.path.join(BASE_DIR, "trainData"))))
    x_train = np.zeros((len(imagePaths), 32, 32, 3))

    for i, path in enumerate(imagePaths):
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        x_train[i] = img_to_array(img)

    y_train = np.arange(len(imagePaths))
    return x_train, y_train


def loadTestingImages1():
    """Split ``testCNN.png`` into eight 32×32 card crops.

    Coordinates are scaled relative to a 1920×1080 reference resolution so
    the helper works on different screen sizes. Raises ``FileNotFoundError``
    if the screenshot does not exist.
    """

    screenshot = os.path.join(BASE_DIR, "testCNN.png")
    if not os.path.exists(screenshot):
        raise FileNotFoundError(
            "testCNN.png not found. Capture the screen before calling "
            "loadTestingImages1()."
        )

    img = cv2.imread(screenshot)
    h, w = img.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0

    # Save the entire row of card slots for debugging
    row = img[int(58 * sy):int(180 * sy), int(702 * sx):int(1230 * sx)]
    cv2.imwrite(os.path.join(BASE_DIR, "croppped.png"), row)

    # Absolute coordinates of each card slot on a 1920×1080 frame
    slots = [
        (115, 203, 752, 806),
        (115, 203, 811, 865),
        (115, 203, 870, 924),
        (115, 203, 929, 983),
        (115, 203, 988, 1042),
        (115, 203, 1047, 1101),
        (115, 203, 1106, 1160),
        (115, 203, 1165, 1219),
    ]

    test_dir = os.path.join(BASE_DIR, "testData")
    for i, (y1, y2, x1, x2) in enumerate(slots, start=1):
        crop = img[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
        cv2.imwrite(os.path.join(test_dir, f"output{i}.png"), crop)

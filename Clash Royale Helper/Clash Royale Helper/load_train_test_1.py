import os
import numpy as np

# Setting up data
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths


def loadTrainingImages1():
    """Load card training images from ``trainData`` folder."""
    imagePaths = sorted(list(paths.list_images("trainData/")))
    x_train = np.zeros((len(imagePaths), 32, 32, 3))

    for i, path in enumerate(imagePaths):
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        x_train[i] = img_to_array(img)

    y_train = np.arange(len(imagePaths))
    return x_train, y_train


def loadTestingImages1():
    """Split ``testCNN.png`` into eight 32×32 card crops.

    Raises a ``FileNotFoundError`` if the screenshot does not exist.
    """
    if not os.path.exists("testCNN.png"):
        raise FileNotFoundError(
            "testCNN.png not found. Capture the screen before calling "
            "loadTestingImages1()."
        )

    img = cv2.imread("testCNN.png")
    arr = img_to_array(img)
    cv2.imwrite("croppped.png", arr[58:180, 702:1230])

    arr = arr[58:180, 702:1230]

    cv2.imwrite("testData/output1.png", arr[57:145, 50:104])
    cv2.imwrite("testData/output2.png", arr[57:145, 109:163])
    cv2.imwrite("testData/output3.png", arr[57:145, 168:222])
    cv2.imwrite("testData/output4.png", arr[57:145, 227:281])
    cv2.imwrite("testData/output5.png", arr[57:145, 286:340])
    cv2.imwrite("testData/output6.png", arr[57:145, 345:399])
    cv2.imwrite("testData/output7.png", arr[57:145, 404:458])
    cv2.imwrite("testData/output8.png", arr[57:145, 463:517])

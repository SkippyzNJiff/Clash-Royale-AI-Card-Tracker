import numpy as np

# Setting up data
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths

from random import randint

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Used for Generating/Labeling Data
from shutil import copyfile
import os
from random import randint

BASE_DIR = os.path.dirname(__file__)

def generateTrainingImages2():

    gen_dir = os.path.join(BASE_DIR, "generatedData")
    currentNumOfData = len(sorted(list(paths.list_images(gen_dir))))

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    i = 0

    while (True):

        if (time.time()-startTime > 1):
            print("--------Captured Data--------")

            im = ImageGrab.grab()
            im.save(os.path.join(gen_dir, f"input{str(i+1+currentNumOfData)}.png"))
            i += 1

            startTime = time.time()

def labelTrainingData2():

    gen_dir = os.path.join(BASE_DIR, "generatedData")
    imagePaths = sorted(list(paths.list_images(gen_dir)))
    train_dir2 = os.path.join(BASE_DIR, "trainData2")
    currentNumOfLabeledData = len(sorted(list(paths.list_images(train_dir2))))

    root = tkinter.Tk()
    myFrame = tkinter.LabelFrame(root, text="Unlabeled Data", labelanchor="n")
    myFrame.pack()

    labeledCount = 0

    for i in range(len(imagePaths)):
        img = Image.open(imagePaths[i])
        img.thumbnail((1500, 1500), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame, image = img)
        panel.image = img
        panel.grid(row=0, column=0)
        root.update()

        label = input()

        if (label != 'e'):
            copyfile(imagePaths[i], os.path.join(train_dir2, label + "input" + str(labeledCount+currentNumOfLabeledData) + ".png"))
            labeledCount += 1

        os.remove(imagePaths[i])

def loadTrainingImages2():

    train_dir2 = os.path.join(BASE_DIR, "trainData2")
    imagePaths = sorted(list(paths.list_images(train_dir2)))
    x_train = np.zeros((len(imagePaths)*2, 28, 28, 3))

    j = 0

    for i in range(len(imagePaths)):

        # Positive Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite("croppped.png", arr[58:88, 702:1215])

        arr = arr[58:88, 702:1215]

        card = int(os.path.basename(imagePaths[i])[0])

        if (card == 0):
            arr = arr[0:30, 50:104]

        elif (card == 1):
            arr = arr[0:30, 109:163]

        elif (card == 2):
            arr = arr[0:30, 168:222]

        elif (card == 3):
            arr = arr[0:30, 227:281]

        elif (card == 4):
            arr = arr[0:30, 286:340]

        elif (card == 5):
            arr = arr[0:30, 345:399]

        elif (card == 6):
            arr = arr[0:30, 404:459]

        elif (card == 7):
            arr = arr[0:30, 464:518]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j] = img

        # Negative Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite("croppped.png", arr[58:88, 702:1215])

        arr = arr[58:88, 702:1215]

        card = int(os.path.basename(imagePaths[i])[0])
        nonPlayedCards = np.arange(8)
        nonPlayedCards = nonPlayedCards.tolist()
        nonPlayedCards.remove(card)

        cardNotPlayed = randint(0, 6)

        if (cardNotPlayed == 0):
            arr = arr[0:30, 50:104]

        elif (cardNotPlayed == 1):
            arr = arr[0:30, 109:163]

        elif (cardNotPlayed == 2):
            arr = arr[0:30, 168:222]

        elif (cardNotPlayed == 3):
            arr = arr[0:30, 227:281]

        elif (cardNotPlayed == 4):
            arr = arr[0:30, 286:340]

        elif (cardNotPlayed == 5):
            arr = arr[0:30, 345:399]

        elif (cardNotPlayed == 6):
            arr = arr[0:30, 404:459]

        elif (cardNotPlayed == 7):
            arr = arr[0:30, 464:518]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j+1] = img

        j += 2

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = (i+1)%2

    return x_train, y_train

def loadTestingImages2():
    """Split ``testCNN.png`` into eight 28×28 elixir/card slots.

    Coordinates are scaled relative to a 1920×1080 reference resolution so
    the helper works on different screen sizes. Raises ``FileNotFoundError``
    if the screenshot does not exist.
    """

    screenshot = os.path.join(BASE_DIR, "testCNN.png")
    if not os.path.exists(screenshot):
        raise FileNotFoundError(
            "testCNN.png not found. Capture the screen before calling "
            "loadTestingImages2()."
        )

    img = cv2.imread(screenshot)
    h, w = img.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0

    # Save row for debugging
    row = img[int(88 * sy):int(118 * sy), int(702 * sx):int(1215 * sx)]
    cv2.imwrite(os.path.join(BASE_DIR, "croppped.png"), row)

    slots = [
        (88, 118, 752, 806),
        (88, 118, 811, 865),
        (88, 118, 870, 924),
        (88, 118, 929, 983),
        (88, 118, 988, 1042),
        (88, 118, 1047, 1101),
        (88, 118, 1106, 1160),
        (88, 118, 1165, 1219),
    ]

    test_dir2 = os.path.join(BASE_DIR, "testData2")
    for i, (y1, y2, x1, x2) in enumerate(slots, start=1):
        crop = img[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
        cv2.imwrite(os.path.join(test_dir2, f"output{i}.png"), crop)

import argparse
import numpy as np

# Training the data
from tensorflow.keras.utils import to_categorical
from LeNetClass import LeNet
# Used for aug data gen
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Used for training
from tensorflow.keras.optimizers import Adam

# Setting up data
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
# Used for predictions

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Use other files
from load_train_test_1 import loadTrainingImages1
from load_train_test_1 import loadTestingImages1

import os

BASE_DIR = os.path.dirname(__file__)


def trainModel1():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImages1()
    # x_test, y_test = loadTestingImages()
    print(x_train.shape)
    print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255
    #x_test /= 255

    y_train = to_categorical(y_train, num_classes=96)
    #y_test = to_categorical(y_test, num_classes=96)


    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = LeNet.build(width=32, height=32, depth=3, classes=96)
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("[INFO] training network...")
    H = model.fit(aug.flow(x_train, y_train, batch_size=BS),
                  validation_data=(x_train, y_train), steps_per_epoch=len(x_train) // BS,
                  epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save_weights(os.path.join(BASE_DIR, "testNet.h5"))


def modelPredicts1():
    """Run inference on the eight cropped card slots in ``testData``."""
    loadTestingImages1()

    train_dir = os.path.join(BASE_DIR, "trainData")
    test_dir = os.path.join(BASE_DIR, "testData")
    imageNames = sorted(list(paths.list_images(train_dir)))
    for i in range(len(imageNames)):
        imageNames[i] = os.path.splitext(os.path.basename(imageNames[i]))[0]

    print("[INFO] loading network...")
    model = LeNet.build(width=32, height=32, depth=3, classes=96)
    model.load_weights(os.path.join(BASE_DIR, "testNet.h5"))

    for i in range(8):
        img = cv2.imread(os.path.join(test_dir, f"output{i+1}.png"))
        orig = img.copy()

        img = cv2.resize(img, (32, 32))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        output = model.predict(img)[0]
        label = output.argmax()

        print(output)
        print(label)

        label = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)
        print(label)

        orig = cv2.resize(orig, (400, 400))
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(0)


def liveModelPredicts1():

    train_dir = os.path.join(BASE_DIR, "trainData")
    imagePaths = sorted(list(paths.list_images(train_dir)))
    imageNames = sorted(list(paths.list_images(train_dir)))

    for i in range(len(imageNames)):
        imageNames[i] = os.path.splitext(os.path.basename(imageNames[i]))[0]

    print("[INFO] loading network...")
    model = LeNet.build(width=32, height=32, depth=3, classes=96)
    model.load_weights(os.path.join(BASE_DIR, "testNet.h5"))

    opponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']
    tempOpponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']

    root = tkinter.Tk()
    myFrame = tkinter.LabelFrame(root, text="Opponent's Cards", labelanchor="n")
    myFrame.pack()

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    while (True):

        if (time.time()-startTime > 1):

            im = ImageGrab.grab()
            im.save(os.path.join(BASE_DIR, "testCNN.png"))
            loadTestingImages1()

            for i in range(8):

                if (opponentCards[i] != "MysteryCard"):
                    continue

                img = cv2.imread(os.path.join(BASE_DIR, "testData", f"output{i+1}.png"))
                img = cv2.resize(img, (32, 32))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model.predict(img)[0]
                label = output.argmax()

                if (imageNames[label] == "MysteryCard"):
                    continue

                elif (tempOpponentCards[i] == imageNames[label]):
                    opponentCards[i] = imageNames[label]

                    img = Image.open(imagePaths[label])
                    img.thumbnail((128, 128), Image.LANCZOS)
                    img = ImageTk.PhotoImage(img)
                    panel = tkinter.Label(myFrame, image = img, borderwidth=10)
                    panel.image = img
                    panel.grid(row=0, column=i)
                    root.update()

                else:
                    tempOpponentCards[i] = imageNames[label]

                labelString = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)

                print(labelString)

            print("--------Opponent's Deck--------")
            print(opponentCards)
            print()
            print()

            startTime = time.time()

def main():
    parser = argparse.ArgumentParser(description="Train or run the card classifier")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "live"],
        default="train",
        help="Operation to perform",
    )
    args = parser.parse_args()

    if args.mode == "train":
        trainModel1()
    elif args.mode == "predict":
        modelPredicts1()
    else:
        liveModelPredicts1()


# --- CNN 1 ---
if __name__ == "__main__":
    main()

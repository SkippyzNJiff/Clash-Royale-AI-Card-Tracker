import numpy as np

import argparse

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
from load_train_test_2 import (
    loadTrainingImages2,
    loadTestingImages2,
    generateTrainingImages2,
    labelTrainingData2,
)

import os

BASE_DIR = os.path.dirname(__file__)

def trainModel2():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImages2()
    print(x_train.shape)
    print(y_train.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255

    y_train = to_categorical(y_train, num_classes=2)

    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("[INFO] training network...")
    H = model.fit(aug.flow(x_train, y_train, batch_size=BS),
                  validation_data=(x_train, y_train), steps_per_epoch=len(x_train) // BS,
                  epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save_weights(os.path.join(BASE_DIR, "testNet2.h5"))

def modelPredicts2():
    """Run inference on the eight cropped elixir/card slots in ``testData2``."""
    loadTestingImages2()

    print("[INFO] loading network...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    model.load_weights(os.path.join(BASE_DIR, "testNet2.h5"))

    for i in range(8):
        img = cv2.imread(os.path.join(BASE_DIR, "testData2", f"output{i+1}.png"))
        orig = img.copy()

        img = cv2.resize(img, (28, 28))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        output = model.predict(img)[0]
        label = output.argmax()
        msg = "Not Placed"

        if label == 1:
            msg = "Placed"

        print(output)
        print(label)

        label = "Card " + str(i) + " - {}: {:.2f}%".format(msg, output[label] * 100)
        print(label)

        orig = cv2.resize(orig, (400, 400))
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(0)

def liveModelPredicts2():

    print("[INFO] loading network...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    model.load_weights(os.path.join(BASE_DIR, "testNet2.h5"))

    opponentHand = ['Card 1', 'Card 2', 'Card 3', 'Card 4', 'Card 5', 'Card 6', 'Card 7', 'Card 8']

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    while (True):

        if (time.time()-startTime > 1):

            im = ImageGrab.grab()
            im.save(os.path.join(BASE_DIR, "testCNN.png"))
            loadTestingImages2()

            for i in range(8):
                img = cv2.imread(os.path.join(BASE_DIR, "testData2", f"output{i+1}.png"))
                img = cv2.resize(img, (28, 28))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model.predict(img)[0]
                label = output.argmax()
                msg = "Not Placed"

                if (label == 1):
                    msg = "Placed"
                    opponentHand.remove("Card " + str(i+1))
                    opponentHand.append("Card " + str(i+1))

                labelString = "Card " + str(i+1) + " - {}: {:.2f}%".format(msg, output[label] * 100)

                print(labelString)

            print("--------Opponent's Hand--------")
            print(opponentHand)
            print()
            print()

            startTime = time.time()

def main():
    parser = argparse.ArgumentParser(description="Train or run the elixir classifier")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "live"],
        default="train",
        help="Operation to perform",
    )
    args = parser.parse_args()

    if args.mode == "train":
        trainModel2()
    elif args.mode == "predict":
        modelPredicts2()
    else:
        liveModelPredicts2()


# --- CNN 2 ---
if __name__ == "__main__":
    main()

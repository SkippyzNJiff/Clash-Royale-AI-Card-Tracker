# Clash Royale AI Card Tracker

Currently, the code is *not* commented very well. This will change soon.

I wrote a report explaining how this AI Assistant was made. It's tailored towards a non-computer-science audience. Please see: [Rough Draft Report](https://github.com/AmarSaini/Clash-Royale-AI-Card-Tracker/blob/master/Clash%20Royale%20Helper/Document/Report.pdf)

Libraries Used:
- OpenCV (Image preprocessing)
- TensorFlow/Keras (Convolutional neural networks)
- TkInter (GUI)
- Pillow (mapping images into GUI)

## Usage

### Installation

Install the required libraries (TensorFlow, OpenCV, Pillow, etc.) with:

```bash
pip install -r requirements.txt
```

### Training and testing

The two training scripts now expose a small command-line interface. By default
they run the training routine; supply `--mode` to switch behaviour:

```bash
# Train the card classifier and save weights to testNet.h5
python "Clash Royale Helper/Clash Royale Helper/train_predict_cards.py" --mode train

# Split a captured screenshot into test crops and run inference
python "Clash Royale Helper/Clash Royale Helper/train_predict_cards.py" --mode predict

# Train the elixir/hand classifier
python "Clash Royale Helper/Clash Royale Helper/train_predict_elixer.py" --mode train

# Run live tracking of the elixir model
python "Clash Royale Helper/Clash Royale Helper/train_predict_elixer.py" --mode live
```

### Full helper

After training (or using the provided `testNet.h5` and `testNet2.h5` weights),
launch the real-time tracker:

```bash
python "Clash Royale Helper/Clash Royale Helper/Clash_Royale_Helper.py"
```

Scripts resolve asset paths relative to their own directory, so you can invoke
them from the project root as shown above.

The cropping utilities scale coordinates based on the captured screenshot's
dimensions (assuming a 16:9 aspect ratio). If your game window uses a very
different layout you may need to tweak the constants inside
``load_train_test_1.py`` and ``load_train_test_2.py``.

If you still want to see the code for the following:

1. The Convolutional Neural Network Architecture (LeNet).

2. Training the first CNN (Convolutional Nerual Network).
3. Predictions on the first CNN.
4. Live (Real-Time) predictions test on the first CNN, to learn opponent's deck.

5. Training the second CNN (Convolutional Nerual Network).
6. Predictions on the second CNN.
7. Live (Real-Time) predictions test on the second CNN, to know which card my opponent is playing.

8. Both CNN's running together during real-time to predict opponent's card cycle + elixir. (This section also has a lot of code regarding the actual AI application).

9. TkInter GUI

The code can be found here: [Clash_Royale_Helper.py](https://github.com/AmarSaini/Clash-Royale-AI-Card-Tracker/blob/master/Clash%20Royale%20Helper/Clash%20Royale%20Helper/Clash_Royale_Helper.py)

![AI Assistant](https://i.imgur.com/r4zqYmj.png)

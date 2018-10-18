# MNIST Classifier in TensorFlow

Just a simple neural network MNIST classifier I built to teach myself. Nothing fancy, no convolutional layers

Consistently gets 98% accuracy or more on the test set with only a couple minutes of training on my computer.

3 layers:

- 2 dense (512 and 256 units, relu activation) and one dropout (with rate of 0.2)

## Requirements:

- Keras (only used for loading the mnist dataset)
- TensorFlow (tested with 1.5.0)
- Numpy
- Matplotlib (visualizes the losses at the end)


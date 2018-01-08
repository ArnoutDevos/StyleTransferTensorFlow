# StyleTransferTensorFlow
Implementation of "A Neural Algorithm of Artistic Style" in TensorFlow

In this notebook we will implement the style transfer technique from the paper "[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)".

The general idea is to take two images, and produce a new image that reflects the content of one but the artistic “style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep neural network, and then performing gradient descent on the pixels of the image itself.

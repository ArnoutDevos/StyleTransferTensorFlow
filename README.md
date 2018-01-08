# StyleTransferTensorFlow
Implementation of the paper "[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576" in TensorFlow

The general idea is to take two images, and produce a new image that reflects the content of one but the artistic “style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep neural network, and then performing gradient descent on the pixels of the image itself.

A more verbose version giving a deeper explanation and going over equations can be found in the following blogpost [A Neural Algorithm of Artistic Style](https://arnoutdevos.github.io/A-Neural-Algorithm-of-Artistic-Style/).

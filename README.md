# StyleTransferTensorFlow
Implementation of the paper "[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)" in TensorFlow

The general idea is to take two images, and produce a new image that reflects the content of one but the artistic “style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep neural network, and then performing gradient descent on the pixels of the image itself.

A more verbose version giving a deeper explanation and going over equations can be found in the following blogpost [A Neural Algorithm of Artistic Style](https://arnoutdevos.github.io/A-Neural-Algorithm-of-Artistic-Style/).

## Usage

### Prerequisites
1. Python 3
2. TensorFlow
3. Python packages : NumPy, SciPy
4. Pretrained VGG19 file : [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Download the file from the link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `model/`

### Running
```
python StyleTransferTensorFlow.py
```

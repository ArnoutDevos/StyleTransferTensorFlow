# Import what we need
#!pip3 install --upgrade scipy
import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf  # Import TensorFlow after Scipy or Scipy will break

# Pick the VGG 19-layer model by from the paper "Very Deep Convolutional 
# Networks for Large-Scale Image Recognition".
VGG_MODEL = 'model/imagenet-vgg-verydeep-19.mat'

# In order to use this VGG model we need to substract the mean of the images
# originally used to train the VGG model from the new input images to be consistent.
# This affects the performance greatly.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,3))

# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image to use.
STYLE_IMAGE = 'images/muse.jpg'
# Content image to use.
CONTENT_IMAGE = 'images/hollywood_sign.jpg'
# Image dimensions constants. 
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
COLOR_CHANNELS = 3

# Layers to use. We will use these layers as advised in the paper.
# To have softer features, increase the weight of the higher layers
# (conv5_1) and decrease the weight of the lower layers (conv1_1).
# To have harder features, decrease the weight of the higher layers
# (conv5_1) and increase the weight of the lower layers (conv1_1).
STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.5),
    ('conv3_1', 0.5),
    ('conv4_1', 0.5),
    ('conv5_1', 0.5),
]

# Constant to put more emphasis on content loss.
ALPHA = 0.0025
# Constant to put more emphasis on style loss.
BETA = 1

def main():
    [_, content_image] = load_image(CONTENT_IMAGE)
    [_, style_image] = load_image(STYLE_IMAGE)

    sess = tf.InteractiveSession()

    # Load VGG model
    model = load_vgg_model(VGG_MODEL)


    print('Loaded VGG model.')

    # Construct content_loss using content_image.
    content_image_list = np.reshape(content_image, ((1,) + content_image.shape))
    sess.run(model['input'].assign(content_image_list))
    content_loss = content_loss_func(sess, model)

    # Construct style_loss using style_image.
    style_image_list = np.reshape(style_image, ((1,) + style_image.shape))
    sess.run(model['input'].assign(style_image_list))
    style_loss = style_loss_func(sess, model)

    # Instantiate equation 7 of the paper.
    total_loss = ALPHA * content_loss + BETA * style_loss

    # We minimize the total_loss, which is the equation 7.
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)

    # Number of iterations to run.
    ITERATIONS = 500

    sess.run(tf.global_variables_initializer())

    gen_image = generate_noise_image(content_image, 0.0)
    input_image = np.reshape(gen_image, ((1,) + gen_image.shape))
    sess.run(model['input'].assign(input_image))
    for it in range(ITERATIONS):
      sess.run(train_step)
      print('Iteration %d' % (it))
      if it % 50 == 0:
        # Print every 50 iterations.
        mixed_image = sess.run(model['input'])
        print('Iteration %d' % (it))
        print('cost: ', sess.run(total_loss))

        if not os.path.exists(OUTPUT_DIR):
          os.mkdir(OUTPUT_DIR)

        filename = 'output/%d.png' % (it)
        save_image(filename, mixed_image[0])

def load_vgg_model(path):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

######

def load_image(path):
    image_raw = scipy.misc.imread(path)
    # Resize the image for convnet input and add an extra dimension
    image_raw = scipy.misc.imresize(image_raw, (IMAGE_HEIGHT, IMAGE_WIDTH))
    # Input to the VGG model expects the mean to be subtracted.
    image = (image_raw - MEAN_VALUES)
    return [image_raw, image]

def recover_image(image):
    image_raw = image + MEAN_VALUES
    image_raw = np.clip(image_raw, 0, 255).astype('uint8')
    return image_raw

def save_image(path, image):
    # Output should add back the mean.
    image = recover_image(image)
    scipy.misc.imsave(path, image)

#######

def generate_noise_image(content_image, noise_ratio):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.randn(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)
    #Take a weighted average of the values
    gen_image = noise_image * noise_ratio + content_image * (1.0 - noise_ratio)
    return gen_image

CONTENT_LAYER = 'conv4_2'

def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(current_feat, content_feat):
        """
        Inputs:
        - current_feat: features of the current image, Tensor with shape [1, height, width, channels]
        - content_feat: features of the content image, Tensor with shape [1, height, width, channels]

        Returns:
        - scalar content loss
        """
        loss = 0.5*tf.reduce_sum(tf.square(current_feat - content_feat))
        return loss
    return _content_loss(sess.run(model[CONTENT_LAYER]), model[CONTENT_LAYER])

########

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(feat):
        """
        Compute the Gram matrix from features.

        Inputs:
        - feat: Tensor of shape (1, H, W, C) giving features for a single image.

        Returns:
        - gram: Tensor of shape (C, C) giving the (optionally normalized) Gram matrices for the input image.
        """
        tensor = feat
        shape = tensor.get_shape()
    
        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)
        return gram
    
    def _style_loss(current_feat, style_feat):
        """
        Inputs:
        - current_feat: features of the current image, Tensor with shape [1, height, width, channels]
        - style_feat: features of the style image, Tensor with shape [1, height, width, channels]

        Returns:
        - scalar style loss
        """
        assert (current_feat.shape == style_feat.shape)
        H = current_feat.shape[1]
        W = current_feat.shape[2]
        M = H * W
        
        N = current_feat.shape[3]
        
        current_feat = tf.convert_to_tensor(current_feat)
        
        gram_current = _gram_matrix(current_feat)
        gram_style = _gram_matrix(style_feat)
        
        loss = 1/(4 * (N ** 2) * (M ** 2)) * tf.reduce_sum(tf.square(gram_current - gram_style))
        return loss

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss

if __name__ == "__main__":
    main()
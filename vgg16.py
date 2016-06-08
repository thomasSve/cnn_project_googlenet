# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer # lasagne.layers.dnn.Conv2DDNNLayer to enforce cuDNN
# from lasagne.layers import Conv2DLayer as ConvLayer # Default convolution, uses cuDNN if available
# from lasagne.layers.cuda_convnet import Conv2DCCLayer # for Krizhevsky's cuda-convnet.
from lasagne.nonlinearities import softmax


def build_model(input_var = None):

    # Each layer is linked to its incoming layer(s), so we only need the output layer(s) to access a network in Lasagne
    
    net = InputLayer((None, 3, 64, 64), input_var = input_var)
    net = ConvLayer(net, 64, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 64, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(net, 128, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 128, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 256, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(net, 512, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = DenseLayer(net, num_units=4096)
    net = DropoutLayer(net, p=0.5)
    net = DenseLayer(net, num_units=4096)
    net = DropoutLayer(net, p=0.5)
    net = DenseLayer(net, num_units=1000, nonlinearity=None)
    net = NonlinearityLayer(net, softmax)

    return net 

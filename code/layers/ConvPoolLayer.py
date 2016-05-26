import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

class ConvPoolLayer():
    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 poolsize, poolstride, bias_init, lrn=False):
        
        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[0]
        self.lrn = lrn

        input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01

        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                kerns=W_shuffled,
                                subsample=(convstride, convstride),
                                border_mode=padsize,
        )
        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

        # ReLu
        self.output = T.maximum(conv_out, 0)

        # Pooling
        if self.poolsize != 1:
            self.output = dnn.dnn_pool(self.output,
                                       ws = (poolsize, poolsize),
                                       stride = (poolstride, poolstride))
            
        self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b


        # LRN
        if self.lrn:
            self.output = self.lrn_func(self.output)

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']


        print "conv layer with shape_in: " + str(image_shape)

        
        

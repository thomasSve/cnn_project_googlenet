import theano
from theano import tensor as T
import numpy as np

### AlexNet Architecture ###

# Max-pooling - Reduce the overall computational 
# Five Convolutional layers, first three with max-pooling
# Two fully connected layers
# One Softmax-layer
# Dropout - To prevent the network to get biased (prevent overfitting) Random dropping neurals.
# 


class AlexNet():
    '''
    The Alexnet paper = https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    Image of network = https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Alexnet.svg/2000px-Alexnet.svg.png
    '''
    def __init__(self, config):
        self.config = config

        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']

        x = T.ftensor4('x')
        y = T.lvector('y')
        
        params, weight_types, rand = self.build_network(x, batch_size, flag_datalayer)

        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size
        
    def build_network(self, x, batch_size, flag_datalayer):
        
        ### BUILD NETWORK ###
        rand = T.fvector('rand')

        print '... building the model'

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256, batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=self.config['rand_crop'])
            layer1_input = data_layer.output

        else:
            layer1_input = x
            
        # Conv1 with norm and pool layer
        conv1 = ConvPoolLayer(input = layer1_input,
                              image_shape = (),
                              filter_shape = ,
                              convstride = , padsize = ,
                              poolsize = , poolstride = ,
                              bias_init = , lrn = True)

        # Conv2 with norm and pool layer
        conv2 = ConvPoolLayer(input = conv1.output,
                              image_shape = (),
                              filter_shape = ,
                              convstride = , padsize = ,
                              poolsize = , poolstride = ,
                              bias_init = , lrn = True)

        # Conv3 without norm and pool layer
        conv3 = ConvPoolLayer(input = conv2.output,
                              image_shape = (),
                              filter_shape = ,
                              convstride = , padsize = ,
                              poolsize = , poolstride = ,
                              bias_init = , lrn = False)
    
        # Conv4 without norm and pool layer
        conv4 = ConvPoolLayer(input = conv3.output,
                              image_shape = (),
                              filter_shape = ,
                              convstride = , padsize = ,
                              poolsize = , poolstride = ,
                              bias_init = , lrn = False)

        # Conv5 with pool layer, without norm 
        conv5 = ConvPoolLayer(input = conv4.output,
                              image_shape = (),
                              filter_shape = ,
                              convstride = , padsize = ,
                              poolsize = , poolstride = ,
                              bias_init = , lrn = False)
        
        # Fully-connected layer
        fc6 = FCLayer(input=fc6_input, n_in=9216, n_out=4096)

        # Layer 6 has dropoutlayer
        drop6 = DropoutLayer(input = fc6.output, n_in=4096, n_out=4096)

        fc7 = FCLayer(input = drop6.output, n_in=9216, n_out=4096)

        # Layer 7 has dropoutlayer
        drop7 = DropoutLayer(input = fc6.output, n_in=4096, n_out=4096)

        softmax8 = SoftmaxLayer(input=drop7.output, n_in=4096, n_out=1000)

        # Merge layers together in an list
        self.layers = [conv1, conv2, conv3, conv4, conv5, fc6, fc7, softmax8]
        
        params = []
        weight_types = []
        for layer in layers:
            params += layer.params
            weight_types += layer.weight_type
            
        ### FINISHED BUILDING NETWORD ###

        self.cost = softmax8.negative_log_likelihood(y)
        self.errors = softmax8.errors(y)
        self.errors_top_5 = softmax8.errors_top_x(y, 5)

        return params, weight_types, rand

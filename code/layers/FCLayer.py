import numpy as np
import theano
import theano.tensor as T

import Weight

class FCLayer():
    def __init__(self, input, n_in, n_out):
        '''
        The fully-connected layer
        '''
        self.w = Weight((n_in, n_out), std = 0.005)
        self.b = Weight(n_out, mean = 0.1, std = 0)

        self.input = input
        lin_output = T.dot(self.input, self.w.val) + self.b.val
        self.output = T.maximum(lin_output, 0)
        self.params = [self.w.val, self.b.val]
        self.weight_type = ['w', 'b']

        print 'fc layer with num_in: ' + str(n_in) + ' num_out: ' str(n_out)
        

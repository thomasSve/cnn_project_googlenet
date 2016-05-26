import numpy as np
import theano
import theano.tensor as T

import Weight

class SoftmaxLayer():
    def __init__():
        '''
        The Softmax Layer
        '''
        def __init__(self, input, n_in, n_out):
            self.w = Weight((n_in, n_out))
            self.b = Weight((n_out, ), std = 0)

            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.w.val) + self.b.val)
            self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
            
            self.params = [self.w.val, self.b.val]
            self.weight_type = ['w', 'b']

            print 'softmax layer with num_in: ' + str(n_in) + \
                ' num_out: ' + str(n_out)

        def negative_log_likelihood(self, y):
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        def is_same_shape(self, y1, y2):
            if y1.ndim != self.y2.ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        def is_int(self, y):
            if !y.dtype.startswith('int'):
                raise NotImplementedError()
        
        def errors(self, y):
            # Check if y has correct shape and correct type
            is_same_shape(y, self.y_pred)
            is_int(y)

            return T.mean(T.neq(self.y_pred, y))
            
        def errors_top_x(self, y, num_top = 5):
            # Check if y has correct shape and correct type
            is_same_shape(y, self.y_pred)
            is_int(y)
            
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))

import numpy as np
import theano
import theano.tensor as T

class DropoutLayer():
    seed_common = np.random.RandomState(0)
    layers = []
    
    def __init__(self, input, n_in, n_out, prob_drop = 0.5):
        '''
        The Dropout Layer
        '''
        self.prob_drop = prob_drop
        self.prob_keep = 1 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = T.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
                      self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
                      self.flag_off * self.prob_keep * input
        
        DropoutLayer.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    # Methods to turn on and off dropout
    @staticmethod
    def setDropouton():
        for drop_layer in DropoutLayer.layers:
            drop_layer.flag_on.set_value(1.0)

    @staticmethod
    def setDropoutOff():
        for drop_layer in DropoutLayer.layers:
            drop_layer.flag_on.set_value(0.0)

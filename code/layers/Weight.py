import numpy as np
import theano
import theano.tensor as T

class Weight():
    
    def __init__(self, w_shape, mean = 0, std = 0.01):
        super(Weight, self).__init__()
        if std != 0:
            self.np_values = np.asarray(rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)
        
    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())
    
    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)

import numpy as np
import theano
import theano.tensor as T


class Datalayer():
    def __init__(self, input, image_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the whole batch
        '''

        # Random mirroring
        mirror = input[:, :, ::-1, :]
        input = T.concatenate([input, mirror], axis = 0)
        
        # Crop image
        center_margin = (image_shape[2] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_x = T.cast(rand[] * center_margin * 2, 'int32')
            crop_y = T.cast(rand[] * center_margin * 2, 'int32')

        print 'data layer with shape_in: ' + str(image_shape)

    

import glob
import time
import os

import numpy as np


from proc_load import crop_and_mirror

def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Create folder: " + config['weights_dir']
        
    return config

def unpack_config(config, ext_data = ".hkl", ext_label = ".npy"):
    flag_para_load = config['para_load']

    # Load Training / Validation Filenames and Labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'val_labels' + ext_label)
    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    return (flag_para_load, train_filenames, val_filenames,
            train_labels, val_labels, img_mean)

def adjust_learning_rate(config, epoch, step_id, val_record, learning_rate):
    # Adapt learning rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_id]:
            learning_rate.set_value(np.float32(leaning_rate.get_value() / 10))
            step_id += 1
            if step_id >= len(config['lr_step']):
                step_id = 0 # Prevent index out of range error
            print 'Learning rate changed to::', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] < config['lr_adapt_threshold']):
            learning_rate.set_value(np.float32(learning_rate.get_value() / 10))
            print 'Learning rate set to:: ' + learning_rate.get_value()

    return step_id

def get_val_error_loss():
    return 0

def getrand3d():
    return 0

def train_model_wrap():
    return 0


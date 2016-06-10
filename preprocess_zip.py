import numpy as np
import h5py
import zipfile
from random import shuffle
from math import floor
from PIL import Image
from StringIO import StringIO

def split_dataset(X, y, test_size = 0.2, val = False):
    print len(X), len(y)
    data = zip(X, y)
    shuffle(data)
    X, y = zip(*data)

    if val:
        split_point = int(floor(len(X)*(1 - test_size * 2)))
        X_train, y_train = X[:split_point], y[:split_point]
        X_test_val, y_test_val = X[split_point:], y[split_point:]
        split_point = int(floor(len(X_test_val)*0.5))
        X_test, y_test, X_val, y_val = X_test_val[:split_point], y_test_val[:split_point], \
                                       X_test_val[split_point:], y_test_val[split_point:]

        return np.array(X_train), np.array(y_train), np.array(X_test)\
            , np.array(y_test), np.array(X_val), np.array(y_val)
    else:
        split_point = int(floor(len(X)*(1 - test_size)))
        return np.array(X[:split_point]), np.array(y[:split_point]), \
                                                  np.array(X[split_point:]), np.array(y[split_point:])


def load_zip_training_set(path, wnids, archive):
    X = []
    y = []
    i = 0
    for class_id in wnids:
        bbox_file = path + class_id + "/" + class_id + "_boxes.txt"
        for line  in archive.open(bbox_file):
            words = line.split()
            img = archive.read(path + class_id + "/images/" + words[0])
            img = Image.open(StringIO(img)) 
            image = np.array(img)
            if image.ndim == 3:
                image = np.rollaxis(image, 2)
                X.append(image) # Append image to dataset
                y.append(i)
        i = i + 1

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.uint8)

def find_label(wnids, wnid):
    i = 0
    for line in wnids:
        if line == wnid:
            return i
        i = i + 1
    return None

def load_zip_val_set(path, wnids, archive):
    X = []
    y = []
    val_annotations = path + "val_annotations.txt"    
    for line in archive.open(val_annotations):
        words = line.split()
        img = archive.read(path + "images/" + words[0])
        img = Image.open(StringIO(img))
        image = np.array(img)
        label = find_label(wnids, words[1])
        if image.ndim == 3 and label != None:
            image = np.rollaxis(image, 2)
            X.append(image) # Append image to dataset
            y.append(find_label(wnids, words[1]))
            
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.uint8)

def generate_url_zip():    
    zip_url = "tiny-imagenet-200.zip"
    train_path = "tiny-imagenet-200/train/"
    val_path = "tiny-imagenet-200/val/"
    test_path = "tiny-imagenet-200/test/"
    wnid_file = "tiny-imagenet-200/wnids.txt"

    
    print "Reading from zip..."
    archive = zipfile.ZipFile(zip_url, 'r') # Open the archive
    wnids = [line.strip() for line in archive.open(wnid_file)] # Load list over classes
    wnids = wnids[:100] # Load only the 100 first classes
    
    print "Loading training set..."
    X, y = load_zip_training_set(train_path, wnids, archive)

    print "Loading validation set.."
    X_val, y_val = load_zip_val_set(val_path, wnids, archive)

    X_train, y_train, X_test, y_test =  split_dataset(X, y, test_size = 0.2)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)    

    return X_train.astype(np.uint8), y_train.astype(np.uint8), X_val.astype(np.uint8), y_val.astype(np.uint8), X_test.astype(np.uint8), y_test.astype(np.uint8)
    
if __name__ == "__main__":
    generate_url_zip()

import numpy as np
import h5py
from random import shuffle
from math import floor

tiny_imagenet = "http://pages.ucsd.edu/~ztu/courses/tiny-imagenet-200.zip"

def crop_image(image, box):
    # xmin, ymin, xmax, ymax
    resized_image = image[int(box[0]):int(box[2]), int(box[1]):int(box[3])]    
    return np.array(resized_image)


def load_dataset():
    with h5py.File('preprocessed_data/train_set.h5','r') as hf:
        X = hf.get('X')
        X_train = np.array(X, dtype=np.uint8)
        y = hf.get('y')
        y_train = np.array(y, dtype=np.uint8)
        
    with h5py.File('preprocessed_data/val_set.h5','r') as hf:
        X = hf.get('X')
        X_val = np.array(X, dtype=np.uint8)
        y = hf.get('y')
        y_val = np.array(y, dtype=np.uint8)

    with h5py.File('preprocessed_data/test_set.h5','r') as hf:
        X = hf.get('X')
        X_test = np.array(X, dtype=np.uint8)
        y = hf.get('y')
        y_test = np.array(y, dtype=np.uint8)
        
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_dataset(filename, X, y = None):
    if y != None:
        with h5py.File("preprocessed_data/" + filename, 'w') as hf:
            hf.create_dataset('X', data=X)
            hf.create_dataset('y', data=y)
    else:
        with h5py.File("preprocessed_data/" + filename, 'w') as hf:
            hf.create_dataset('X', data=X)

def load_training_set(path, Image, wnids):
    import glob, os
    owd = os.getcwd() # Get original path

    images = []
    y = []
    bbox = []
    i = 0
    
    for class_id in wnids:
        bbox_file = path + class_id + "/" + class_id + "_boxes.txt"
        for line  in open(bbox_file):
            words = line.split()
            #img = Image.open(path + class_id + "/images/" + words[0]).convert('L')
            img = Image.open(path + class_id + "/images/" + words[0])

            image = np.array(img)
            #image_cropped = crop_image(image, words[1:])
            if image.ndim == 3:
                bbox.append(words[1:])
                #image = np.ravel(image)  #Reshape image into columnvector
                image = np.rollaxis(image, 2)
                images.append(image) # Append image to dataset
                y.append(i)
        i = i + 1
        os.chdir(owd) # Reset to original path

    X = np.array(images, dtype=np.uint8)
    y = np.array(y)
    bbox = np.array(bbox)
        
    return X, y, bbox

def load_val_set(path, Image):
    val_annotations = path + "val_annotations.txt"
    images_path = path + "images/"

    images = []
    y = []
    bbox = []
    
    for line in open(val_annotations):
        words = line.split()
        image_file = words[0]
        #img = Image.open(images_path + image_file).convert('L') # Read image and convert to grayscale
        img = Image.open(images_path + image_file)
        image = np.array(img)
        #image_cropped = crop_image(image, words[2:])

        #image = np.ravel(image) # Convert the image to a columnvector
        #print image_file, image.shape
        if image.ndim == 3:
            y.append(words[1])
            bbox.append(words[2:])
            image = np.rollaxis(image, 2)
            images.append(image)


    X = np.array(images, dtype=np.uint8)
    y = np.array(y)
    bbox = np.array(bbox)
    
    return X, y, bbox

def load_test_set(test_path):
    import glob, os
    owd = os.getcwd() # Get original path
    images = []
    
    for file in glob.glob("*.JPEG"): # For all images in folder
        img = Image.open(file)
        image = np.array(img)
        image = np.rollaxis(image, 2)
        images.append(image)
    os.chdir(owd) # Reset to original path
    
    return np.array(images)

def split_dataset(X, y, test_size = 0.2, val = False):
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
        return X[:split_point], y[:split_point], X[split_point:], y[split_point:]
        
def generate_dataset(num_classes = 200, save = True):
    import Image
    print("Generating dataset...")
    train_path = "/home/thomas/data/dataset/tiny-imagenet-200/train/"
    val_path = "/home/thomas/data/dataset/tiny-imagenet-200/val/"
    test_path = "/home/thomas/data/dataset/tiny-imagenet-200/test/"
    wnid_file = "/home/thomas/data/dataset/tiny-imagenet-200/wnids.txt"

    wnids = [line.strip() for line in open(wnid_file)]
    wnids = wnids[:num_classes]
    print "Classes: ", len(wnids)
    print "Loading training set"
    X_train, y_train, train_box = load_training_set(train_path, Image, wnids)

    print "Loading validation set"
    if num_classes == 200:
        X, y, train_box = load_training_set(train_path, Image, wnids)
        X_train, y_train, X_test, y_test = split_dataset(X, y, test_size = 0.2)
        X_val, y_val, val_box = load_val_set(val_path, Image)
    else:
        X, y, boxes = load_training_set(train_path, Image, wnids)
        X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(X, y, test_size = 0.15, val = True)
    print "X_val shape: ", X_val.shape, " y_val shape: ", y_val.shape
    print "X_train shape: ", X_train.shape, " y_train shape: ", y_train.shape
    print "X_test shape: ", X_test.shape, " y_test shape: ", y_test.shape

    if save:
        save_dataset("train_set.h5", X_train, y_train)
        save_dataset("val_set.h5", X_val, y_val)
        save_dataset("test_set.h5", X_test, y_test)
        print("Dataset saved")
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
    

if __name__ == "__main__":
    generate_dataset(20)
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    #print X_train.shape, y_train.shape
    #print X_val, y_val

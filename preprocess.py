import numpy as np
from scipy.misc import imread, imsave, imresize

train_anns_path = 'imagenet/class+loc/train/annotations'
train_image_dir = 'imagenet/class+loc/train/images'
val_anns_path = 'imagenet/class+loc/val/annotations'
val_image_dir = 'imagenet/class+loc/val/images'


def crop_image(image, box):
    
    xmin, ymin, xmax, ymax = box[0], box[2], box[1], box[3]    
    resized_image = image[int(xmin):int(xmax), int(ymin):int(ymax)]

    print len(resized_image)
    
    return resized_image


def load_dataset():
    train_set = np.load("preprocessed_data/train_set.npz")
    X_train, y_train = train_set['X'], train_set['y']
    
    val_set = np.load("preprocessed_data/val_set.npz")
    X_val, y_val = val_set['X'], val_set['y']
    
    test_set = np.load("preprocessed_data/test_set.npz")
    X_test, y_test = test_set['X']
    
    return X_train, y_train, X_val, y_val

def save_dataset(filename, X, y = None):
    if y != None:
        np.savez("preprocessed_data/" + filename, X=X, y=y)
    else:
        np.savez("preprocessed_data/" + filename, X=X)
          
def load_training_set(path, wnids):
    import glob, os
    owd = os.getcwd() # Get original path

    images = []
    y = []
    bbox = []
    for class_id in wnids:
        bbox_file = path + class_id + "/" + class_id + "_boxes.txt"
        bbox.append(line.strip for line in open(bbox_file))
        os.chdir(path + class_id + "/images/") # Change path to subfolder

        for file in glob.glob("*.JPEG"): # For all images in folder
            image = imread(file) # Read image to numpy array
            image = np.ravel(image)  #Reshape image into columnvector
            images.append(image) # Append image to dataset
            y.append(class_id)
        os.chdir(owd) # Reset to original path

    return images, y, bbox

def load_val_set(path):
    val_annotations = path + "val_annotations.txt"
    images_path = path + "images/"

    images = []
    y = []
    bbox = []
    
    for line in open(val_annotations):
        words = line.split()
        image_file = words[0]
        image = np.array(imread(images_path + image_file))
        y.append(words[1])
        bbox.append(words[2:])
        
        cropped_image = crop_image(image, words[2:])
        
        images.append(np.ravel(image))

    return images, y, bbox

def load_test_set(test_path):
    import glob, os
    owd = os.getcwd() # Get original path
    images = []
    
    for file in glob.glob("*.JPEG"): # For all images in folder
        image = imread(file)
        images.append(np.ravel(image))
    os.chdir(owd) # Reset to original path
    
    return images

def generate_dataset():
    print("Generating dataset...")
    train_path = "/home/thomas/data/dataset/tiny-imagenet-200/train/"
    val_path = "/home/thomas/data/dataset/tiny-imagenet-200/val/"
    test_path = "/home/thomas/data/dataset/tiny-imagenet-200/test/"
    wnid_file = "/home/thomas/data/dataset/tiny-imagenet-200/wnids.txt"

    wnids = [line.strip() for line in open(wnid_file)]
    print len(wnids)
    
    #X_train, y_train, train_box = load_training_set(train_path, wnids)
    X_val, y_val, val_box = load_val_set(val_path)
    #test_set = load_test_set(test_path)

    # crop_images(X_train, train_box)
    crop_images(X_val, val_box)
    
    # Save the generated arrays
    print("Saving dataset...")
    save_dataset("train_set", X_train, y_train)
    save_dataset("val_set", X_val, y_val)
    #save_dataset("test_set", test_set['X'])
    print("Dataset saved")


if __name__ == "__main__":
    generate_dataset()

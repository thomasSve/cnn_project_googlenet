import numpy as np
import lasagne
import googlenet
import zipfile
import theano
from random import shuffle
from math import floor
from preprocess_zip import load_zip_val_set
from PIL import Image
from StringIO import StringIO


def load_pickle_googlenet():
    import pickle
    model = pickle.load(open('vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']
    lasagne.layers.set_all_param_values(output_layer, model['values'])
    
    return output_layer

def load_network():
    network = googlenet.build_model()
    with np.load('trained_googlenet_100.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    return network

def load_test_images(image_urls):
    images = []
    images_raw = []
    
    for url in image_urls:
        img = Image.open(url)
        image = np.array(img)
        images_raw.append(image)
        image = np.rollaxis(image)
        images.append(image)
        
    return images, images_raw

def load_images(path, wnids, archive):
    X = []
    X_raw = []
    val_annotations = path + "val_annotations.txt"    
    for line in archive.open(val_annotations):
        words = line.split()
        img = archive.read(path + "images/" + words[0])
        img = Image.open(StringIO(img))
        image = np.array(img)
        if image.ndim == 3:
            X_raw.append(image)
            image = np.rollaxis(image, 2)
            X.append(image) # Append image to dataset
            
    return np.array(X), np.array(X_raw)

def random_test_images(image_urls, num_samples = 5):
    np.random.seed(23)
    image_urls = image_urls[:num_samples]
    images, images_raw = load_test_images(image_urls)
    
    return images, images_raw

def load_classes_name(wnids, archive):
    words = "tiny-imagenet-200/words.txt"
    classes_words = {}
    for line in archive.open(words):
        words = line.split()
        classes_words[words[0]] = words[1]

    return classes_words

def load_classes(wnid_file, archive):
    return [line.strip() for line in archive.open(wnid_file)]

def save_predictions(images, images_raw, network, classes, classes_words):
    top5 = []
    for i in range(len(images)):
            prob = np.array(lasagne.layers.get_output(network, images[i].astype(theano.config.floatX), deterministic=True).eval())
            top5.append(np.argsort(prob[0])[-1:-6:-1])

    np.savez("predictions.npz", top=top5, images=images, images_raw=images_raw, classes=classes, classes_words=classes_words) 
    
                
def main():
    #with np.load('googlenet_epochs.npz') as data:
    #    results = data['results']

    #epoch_print = [1, 5, 10, 20, 30, 50, 100, 150, 200, 250]
    #for i in epoch_print:
    #    print "Results, epoch " + str(i) + ": " + str(results[i - 1])

        
    zip_url = "tiny-imagenet-200.zip"
    wnid_file = "tiny-imagenet-200/wnids.txt"
    test_path = "tiny-imagenet-200/test/"
    val_path = "tiny-imagenet-200/val/"


    print "Reading from zip..."
    archive = zipfile.ZipFile(zip_url, 'r')
    wnids = [line.strip() for line in archive.open(wnid_file)] # Load list over classes
    wnids = wnids[:100] # Load only the 100 first classes
    classes_words = load_classes_name(wnids, archive)
    
    network = load_network()
    #network = load_pickle_googlenet()
    #images, images_raw = random_test_images()
    X, X_raw = load_images(val_path, wnids, archive)

    data = zip(X, X_raw)
    np.random.shuffle(data)
    data = data[:5]
    X, X_raw = zip(*data)
        
    classes = load_classes(wnid_file, archive)
    save_predictions(X, X_raw, network, classes, classes_words)


    
if __name__=="__main__":
    main()

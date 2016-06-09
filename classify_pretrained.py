import numpy as np
import matplotlib.pyplot as plt
import lasagne
import googlenet

def load_pickle_googlenet():
    import pickle
    model = pickle.load(open('vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']
    lasagne.layers.set_all_param_values(output_layer, model['values'])

    return output_layer

def load_network():
    network = googlenet.build_model()
    with np.load('trained_alexnet_200.npz') as f:
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
        
def random_test_images(image_urls, num_samples = 5):
    np.random.seed(23)
    image_urls = image_urls[:num_samples]
    images, images_raw = load_test_images(image_urls)
    return images, images_raw

def load_classes(wnid_file):
    return [line.strip() for line in open(wnid_file)]

def print_predictions(images, images_raw, network, classes):
    for image, image_raw in images, images_raw:
            prob = np.array(lasagne.layers.get_output(network, image, deterministic=True).eval())
            top5 = np.argsort(prob[0])[-1:-6:-1]

            plt.figure()
            plt.imread(image_raw.astype('uint8'))
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, classes[label]), fontsize=14)

            plt.save("predicted_" + str(i) + ".JPEG")
            i = i + 1
                
def main():
    wnid_file = "/home/thomas/data/dataset/tiny-imagenet-200/wnids.txt"
    
    network = load_network()
    #network = load_pickle_googlenet()
    images, images_raw = random_test_images()
    classes = load_classes(wnid_file)
    print_predictions(images, images_raw, network, classes)

if __name__=="__main__":
    main()

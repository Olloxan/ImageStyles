import numpy as np
import PIL
import random

import time
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array

from itertools import islice
from tqdm import tqdm
import scipy

import matplotlib.pyplot as plt
from scipy import ndimage


def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[-3], x.shape[-2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalizeL2(img):    
    return img / (K.sqrt(K.mean(K.square(img))) + 1e-5)

def showarray(imgArray, format='png'):    
    plt.imshow(imgArray)
    plt.show()
    

def visstd(imgArray, s=0.1):
    '''Normalize and clip the image range for visualization'''
    imgArray = (imgArray - imgArray.mean()) / max(imgArray.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(imgArray, 0, 1) * 255)


model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print('Model loaded.')

model.summary()

input_img = model.input
neuron_index = 7

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'block5_conv1'
# we build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
if K.image_data_format() == 'channels_first':
    loss = K.mean(layer_output[:, neuron_index, :, :])
else:
    loss = K.mean(layer_output[:, :, :, neuron_index])

# we compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads = normalizeL2(grads)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

# step size for gradient ascent
step = 1.

# we start from a gray image with some random noise
if K.image_data_format() == 'channels_first':
    img_data = np.random.uniform(size=(1, 3, 256, 256, 3)) + 128.
else:
    img_data = np.random.uniform(size=(1, 256, 256, 3)) + 128.


# we run gradient ascent for 20 steps
for i in range(50):
    loss_value, grads_value = iterate([img_data])
    img_data += grads_value * step

showarray(visstd(img_data[0]))

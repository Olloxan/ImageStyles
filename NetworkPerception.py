import numpy as np

import keras.backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array, save_img

import scipy

from scipy import ndimage
import matplotlib.pyplot as plt
from utility import displayImageION, showImage


def visstd(imgArray, s=0.1):
    '''Normalize and clip the image range for visualization'''
    imgArray = (imgArray - imgArray.mean()) / max(imgArray.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(imgArray, 0, 1) * 255)

def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1,1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)

def normalizeL2(img):    
    return img / (K.sqrt(K.mean(K.square(img))) + 1e-5)

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

# --- functions---

model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

input_img = model.input

settings = {
    'block3_pool':0.1,
    'block4_pool':1.2,
    'block5_pool':1.5
    }
loss = K.variable(0.)

for layer_name, coeff in settings.items():
    layer_output = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(layer_output), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss = loss + coeff * K.sum(K.square(layer_output[:,:,2:-2,2:-2])) / scaling
    else:
        loss = loss + coeff * K.sum(K.square(layer_output[:,2:-2,2:-2,:])) / scaling

grads = K.gradients(loss, input_img)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
iterate = K.function([input_img], [loss, grads])

num_octave = 5 # Number of scales at which to run gradient ascent
octave_scale = 1.2 # Size ratio between scales

img = preprocess_image('data/bluewaters.jpg')

if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [tuple(int(dim / (octave_scale ** i)) for dim in original_shape) for i in range(num_octave - 1, -1, -1)]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('processing image shape', shape)
    img = resize_img(img, shape)
    for i in range(20):
        loss_value, grads_value = iterate([img])
        img += grads_value * 0.1
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    displayImageION(visstd(img[0]))

showImage(visstd(img[0]))

x=5


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
#from scipy.misc import imresize
#You can use skimage.transform.resize instead
from scipy import ndimage

from io import BytesIO

from IPython.display import clear_output, Image, display, HTML

import matplotlib.pyplot as plt
#plt.ion()

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(img):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        img = img.reshape((3,img.shape[2], img.shape[3]))
        img = img.transpose((1,2,0))
    else:
        img = img.reshape((img.shape[-3], img.shape[-2], 3))
    img /= 2.
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def normalizeL2(tensor):    
    return tensor /(K.sqrt(K.mean(K.square(tensor))) + 1e-5)

def showarray(array, format='png'):
    #buffer=BytesIO()
    #PIL.Image.fromarray(array).save(buffer, format)
    plt.imshow(array)
    plt.show()
    #display(Image(data=buffer.getvalue()))

def visstd(img, s=0.1):
    img = (img - img.mean()) / max(img.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(img, 0,1) * 255)

model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print('model loaded')

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

#  step size for gradient ascent
step = 1.

# we start from a gray image with some random noise
if K.image_data_format() == 'channels_first':
    img_data = np.random.uniform(size(1,3, 256, 256, 3)) + 128
else:
    img_data = np.random.uniform(size=(1,256,256,3)) + 128

# we run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([img_data])
    img_data += grads_value * step

showarray(visstd(img_data[0]))

sample_size = 100
grid = []
layers = [layer_dict['block%d_conv%d' % (i, (i+2) // 3)] for i in range (1,6)]

for layer in layers:
    row = []
    neurons = list(range(max(x or 0 for x in layer.output_shape)))
    if len(neurons) > sample_size:
        neurons = random.sample(neurons, sample_size)
    for neuron in tqdm(neurons, desc=layer.name):
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer.output[:, neuron, :, :])
        else:
            loss = K.mean(layer.output[:, :, :, neuron])
        grads = K.gradients(loss, input_img)[0]
        grads = normalizeL2(grads)
        iterate = K.function([input_img], [loss, grads])
        if K.image_data_format() == 'channels_first':
            img_data = np.random.uniform(size=(1, 3, 128, 128, 3)) + 128.
        else:
            img_data = np.random.uniform(size=(1, 128, 128, 3)) + 128.
        
        for i in range(25):
            loss_value, grads_value = iterate([img_data])
            img_data += grads_value
        row.append((loss_value, img_data[0]))
    grid.append([cell[1] for cell in islice(sorted(row, key=lambda t: -t[0]), 10)])

cols = 10
img_grid = PIL.Image.new('RGB', (cols * 100 + 4, len(layers) * 100 + 4), (180, 180, 180))
for y in range(len(layers)):
    for x in range(cols):
        sub = PIL.Image.fromarray(visstd(grid[y][x])).crop((16, 16, 112, 112))
        img_grid.paste(sub, (x * 100 + 4, (y * 100) + 4))
display(img_grid)
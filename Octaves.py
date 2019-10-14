import numpy as np

import keras.backend as K
from keras.applications import vgg16

import scipy

from scipy import ndimage
import matplotlib.pyplot as plt

#from keras.backend.tensorflow_backend import set_session
#import tensorflow as tf
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

#sess = tf.compat.v1.Session(config=config)

#set_session(sess)


plt.ion()

image = np.zeros((64, 64,3))
fig, ax = plt.subplots()
im = ax.imshow(image)

def displayImage(image):        
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(0.1)

def showarray(imgArray, format='png'):    
    plt.imshow(imgArray)
    plt.show()   

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

model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

input_img = model.input

if K.image_data_format() == 'channels_first':
    img_data = np.random.uniform(size=(3, 64, 64, 3)) + 128.
else:
    img_data = np.random.uniform(size=(64, 64, 3)) + 128.

#showarray(visstd(img_data))

img2 = resize_img([img_data], (256,256))

#showarray(visstd(img2[0]))

size = 64

if K.image_data_format() == 'channels_first':
    img_data = np.random.uniform(size=(1, 3, size, size)) + 128.
else:
    img_data = np.random.uniform(size=(1, size, size, 3)) + 128.

layer = layer_dict['block5_conv1']

neuron = 4 #7


if K.image_data_format() == 'channels_first':    
    loss = K.mean(layer.output[:, neuron, :, :])
else:
    test = layer.output[:, :, :, neuron]
    loss = K.mean(layer.output[:, :, :, neuron])

grads = K.gradients(loss, input_img)[0]
grads = normalizeL2(grads)

iterate = K.function([input_img], [loss, grads])


for octave in range(30):
    if octave > 0:
        size = int(size * 1.1)
        img_data = resize_img(img_data, (size,size))
    for i in range(10):
        loss_value, grads_value = iterate([img_data])

        sigma = (1-octave * 0.03) * 3
        if K.image_data_format() == 'channels_first':
            grads_value[0, 0, :, :] = ndimage.gaussian_filter(grads_value[0, 0, :, :, 0], sigma=sigma)
            grads_value[0, 1, :, :] = ndimage.gaussian_filter(grads_value[0, 1, :, :, 0], sigma=sigma)
            grads_value[0, 2, :, :] = ndimage.gaussian_filter(grads_value[0, 2, :, :, 0], sigma=sigma)
        else:
            grads_value[0, :, :, 0] = ndimage.gaussian_filter(grads_value[0, :, :, 0], sigma=sigma)
            grads_value[0, :, :, 1] = ndimage.gaussian_filter(grads_value[0, :, :, 1], sigma=sigma)
            grads_value[0, :, :, 2] = ndimage.gaussian_filter(grads_value[0, :, :, 2], sigma=sigma)
        img_data += grads_value

    displayImage(visstd(img_data[0]))
    print(octave)

<<<<<<< HEAD

showarray(visstd(img_data[0]))
=======
showarray(visstd(img_data[0]))
x=5
>>>>>>> ff59ed537e0d0aff5c1fa35580cc5d284e83021c

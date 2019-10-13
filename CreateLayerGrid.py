import numpy as np
import PIL
import random

from keras.applications import vgg16
from keras import backend as K
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalizeL2(img):    
    return img / (K.sqrt(K.mean(K.square(img))) + 1e-5)

def showarray(imgArray, format='png'):  
    plt.imshow(imgArray)
    plt.show()

model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

input_img = model.input


sample_size = 100
grid = []
layers = [layer_dict['block%d_conv%d' % (i, (i + 2) // 3)] for i in range(1, 6)]
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
        #for i in range(25):
        for i in range(1):
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
showarray(img_grid)

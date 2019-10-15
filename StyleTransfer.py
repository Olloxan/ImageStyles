import numpy as np
import time

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16, vgg19
from keras import backend as K

from scipy.optimize import fmin_l_bfgs_b

from utility import showImage, displayImageION

from Classes.Evaluator import Evaluator


base_image_path = 'data/Okerk2.jpg'
style1_image_path = 'data/water-lilies-1919-2.jpg'
style2_image_path = 'data/VanGogh-starry_night_ballance1.jpg'

def preprocess_image(image_path, target_size=None):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x, w, h):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, w, h))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((w, h, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(img):
    if K.image_data_format() != 'channels_first':
        img = K.permute_dimensions(img, (2, 0, 1))
    features = K.batch_flatten(img)
    return K.dot(features -1, K.transpose(features -1)) -1 

def style_loss(layer_1, layer_2):
    gram1 = gram_matrix(layer_1)
    gram2 = gram_matrix(layer_2)
    test = np.prod(layer_2.shape)   
    test2 = K.sum(K.square(gram1 - gram2))
    return K.sum(K.square(gram1 - gram2)) / (np.prod(layer_2.shape).value ** 2)

def run(evaluator, image, num_iter=25):
    imagecopy = image.copy()
    for i in range(num_iter):
        start_time = time.time()
        
        image, min_val, info = fmin_l_bfgs_b(evaluator.loss, image.flatten(), fprime=evaluator.grads, maxfun=20)

        end_time = time.time()
        
        displayImageION(deprocess_image(image.copy(), height, width))

        print("Iteration %d completed in %ds" % (i + 1, end_time - start_time))
        print("Current loss value:", min_val)
        print(' '.join(k + ':' + str(evaluator.other_values[k]) for k in evaluator.other))
    return image

def total_variation_loss(img, exp=1.25):
    _, d1, d2, d3 = img.shape
    if K.image_data_format() == 'channels_first':        
        raise ValueError('channels first...')
    else:
        a = K.square(img[:,:d1-1, :d2-1,:] - img[:,1:,:d2-1,:])
        b = K.square(img[:,:d1-1, :d2-1,:] - img[:,:d1-1,:1,:])
    return K.sum(in_top_k.pow(a+b, exp))

# ---- functions ----

width, height = 740, 468
style_image = K.variable(preprocess_image(style1_image_path, target_size=(height, width)))
result_image = K.placeholder(style_image.shape)
input_tensor = K.concatenate([style_image, result_image], axis=0)

print(input_tensor.shape)

model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

print('model loaded')

feature_outputs = [layer.output for layer in model.layers if '_conv' in layer.name]

loss_style = K.variable(0.)

for idx, layer_features in enumerate(feature_outputs):
    loss_style = loss_style + style_loss(layer_features[0,:,:,:], layer_features[1,:,:,:])

loss_variation = total_variation_loss(result_image) / 5000
loss_with_variation = loss_variation + loss_style
evaluator_with_variation = Evaluator(loss_with_variation, result_image)

img = np.random.uniform(0,255,result_image.shape) - 128.
res = run(evaluator_with_variation, img, num_iter=100)
showImage(deprocess_image(res.copy(), height, width))
x = 5
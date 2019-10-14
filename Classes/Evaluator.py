import numpy as np
from keras import backend as K


class Evaluator(object):
    def __init__(self, loss_total, result_image, **other):
        grads = K.gradients(loss_total, result_image)
        outputs = [loss_total] + list(other.values()) + grads
        self.iterate = K.function([result_image], outputs)
        self.other = list(other.keys())
        self.other_values = {}
        self.shape = result_image.shape

        self.loss_value = None
        self.grads_values = None

    def loss(self, img):
        outs = self.iterate([img.reshape(self.shape)])
        self.loss_value = outs[0]
        self.grad_values = outs[-1].flatten().astype('float64')
        self.other_values = dict(zip(self.other, outs[1:-1]))
        return self.loss_value

    def grads(self, img):
        return np.copy(self.grad_values)
       

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
from scipy.misc import imresize
from scipy import ndimage

from io import BytesIO



from IPython.display import clear_output, Image, display, HTML

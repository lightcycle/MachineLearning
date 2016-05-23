from re import match
from scipy.ndimage import imread
from scipy.misc import imresize
from random import shuffle
from functools import partial


def get_label(filepath):
    labels = {"1000": [1,0,0,0], "0100": [0,1,0,0], "0010": [0,0,1,0], "0001": [0,0,0,1]}
    return labels[match(r".*-(.*)\..*", filepath).group(1)]


def load_image(filepath, scale = None):
    image = imread(filepath)
    if scale is not None:
        image = imresize(image, scale)
    return image


def load_dataset(filename_list, batch_size = None, scale = None):
    if batch_size is not None:
        shuffle(filename_list)
        filename_list = filename_list[:batch_size]
    return map(partial(load_image, scale = scale), filename_list), map(get_label, filename_list)
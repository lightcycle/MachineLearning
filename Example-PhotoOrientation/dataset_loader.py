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


def get_dataset_batcher(filename_list, batch_size, scale = None, repeat = True):
    shuffle(filename_list)
    while True:
        for filelist_batch in [filename_list[i:i + batch_size] for i in range(0, len(filename_list), batch_size)]:
            yield map(partial(load_image, scale = scale), filelist_batch), map(get_label, filelist_batch)
        if not repeat:
            break;
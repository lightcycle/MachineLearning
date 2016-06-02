import urllib2
import StringIO
import os
import traceback
import logging
import random
from PIL import Image


def random_image_url():
    print 'Loading URL index'
    index_file = open('images.lst', 'r')
    urls = index_file.readlines()
    print 'Shuffling URL index'
    random.shuffle(urls)
    for url in urls:
        yield url


def random_image():
    for url in random_image_url():
        data = urllib2.urlopen(url).read()
        image = Image.open(StringIO.StringIO(data))
        yield image


def usable_image(image):
    return image.mode == "RGB" and image.size >= output_size and image.getcolors(mincolors) is None


def random_rotation():
    rotation_labels = {0: '1000', 90: '0100', 180: '0010', 270: '0001'}
    rotation = random.randint(0, 3) * 90
    return rotation, rotation_labels[rotation]


def get_center_crop(image):
    width, height = image.size

    if width > height:
        delta = width - height
        left = int(delta / 2)
        upper = 0
        right = height + left
        lower = height
    else:
        delta = height - width
        left = 0
        upper = int(delta / 2)
        right = width
        lower = width + upper
    return left, upper, right, lower


def get_formatted_image(image):
    image = image.crop(get_center_crop(image))
    image.thumbnail(output_size)
    return image


def create_dataset(dir, num):
    if not os.path.exists(dir):
        os.makedirs(dir)

    images = random_image()
    i = 0
    while len(os.listdir(dir)) < num:
        try:
            image = next(images)
            if usable_image(image):
                angle, label = random_rotation()
                path = os.path.join(dir, str(i) + "-" + label + ".jpg")
                get_formatted_image(image).rotate(angle).save(path)
                i += 1
        except StopIteration:
            images = random_image()
        except Exception as e:
            logging.error(traceback.format_exc())


mincolors = 10000
output_size = (200, 200)
random.seed()

create_dataset("./train", 100000)
create_dataset("./test", 10000)

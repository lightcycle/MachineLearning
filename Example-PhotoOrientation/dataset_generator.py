import cStringIO
import os
import traceback
import logging
import tempfile
import random
import mwclient
from hashlib import md5
from PIL import Image


def random_unique_image_title():
    for title in (page['title'] for page in site.random('6', limit = None)):
        if title.lower().endswith((".png", "jpg")):
            hash = md5(title.encode('utf-8')).hexdigest()
            if hash in hashes:
                continue
            hashes.add(hash)
            yield title


def random_image():
    for title in random_unique_image_title():
        data = cStringIO.StringIO()
        mwclient.image.Image(site, title).download(data)
        yield Image.open(data)


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
    while len(os.listdir(dir)) < num:
        try:
            image = next(images)
            if usable_image(image):
                angle, label = random_rotation()
                tf = tempfile.NamedTemporaryFile(suffix = "-" + label + ".jpg", prefix = "", dir = dir, delete = False)
                get_formatted_image(image).rotate(angle).save(tf.name)
        except StopIteration:
            images = random_image()
        except Exception as e:
            logging.error(traceback.format_exc())


site = mwclient.client.Site('commons.wikimedia.org')
mincolors = 10000
output_size = (200, 200)
hashes = set()
random.seed()

create_dataset("./train", 10000)
create_dataset("./test", 10000)
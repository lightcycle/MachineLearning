import cStringIO
import os
import traceback
import logging
import tempfile
import random
import mwclient
from PIL import Image


def random_image_title():
    for title in (page['title'] for page in site.random('6', limit = None)):
        if title.lower().endswith((".png", "jpg")):
            yield title


def random_image():
    for title in random_image_title():
        data = cStringIO.StringIO()
        mwclient.image.Image(site, title).download(data)
        yield Image.open(data)


def usable_image(image):
    return image.mode == "RGB" and image.size >= output_size and image.getcolors(mincolors) is None


def random_rotation():
    rotation_labels = {0: '1000', 90: '0100', 180: '0010', 270: '0001'}
    rotation = random.randint(0, 3) * 90
    return rotation, rotation_labels[rotation]


def get_formatted_image(image):
    image.thumbnail(output_size)
    new_image = Image.new("RGB", output_size)
    new_image.paste(image, ((new_image.size[0] - image.size[0]) / 2, (new_image.size[1] - image.size[1]) / 2))
    return new_image


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
output_size = (400, 400)
random.seed()

create_dataset("./train", 10000)
create_dataset("./test", 10000)
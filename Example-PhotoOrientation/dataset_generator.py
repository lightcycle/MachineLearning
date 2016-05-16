import wikipedia
import urllib
import cStringIO
import os
import traceback
import logging
import tempfile
import random
from PIL import Image

mincolors = 1024
output_size = (400, 400)


def random_image_urls():
    page = wikipedia.page(wikipedia.random())
    return [url for url in page.images if url.endswith(".png") or url.endswith(".jpg")]


def load_image(url):
    file = cStringIO.StringIO(urllib.urlopen(url).read())
    return Image.open(file)


def random_rotation():
    rotation_labels = {0: '1000', 90: '0100', 180: '0010', 270: '0001'}
    rotation = random.randint(0, 3) * 90
    return rotation, rotation_labels[rotation]


def usable_image(image):
    return image.mode == "RGB" and image.size >= output_size and image.getcolors(mincolors) is None


def get_formatted_image(image):
    image.thumbnail(output_size)
    new_image = Image.new("RGB", output_size)
    new_image.paste(image, ((new_image.size[0] - image.size[0]) / 2, (new_image.size[1] - image.size[1]) / 2))
    return new_image


def create_dataset(dir, num):
    if not os.path.exists(dir):
        os.makedirs(dir)

    while num > 0:
        try:
            for url in random_image_urls():
                image = load_image(url)
                if usable_image(image):
                    angle, label = random_rotation()
                    tf = tempfile.NamedTemporaryFile(suffix = "-" + label + ".jpg", prefix = "", dir = dir, delete = False)
                    get_formatted_image(image).rotate(angle).save(tf.name)
                    num -= 1
        except Exception as e:
            logging.error(traceback.format_exc())


random.seed()
wikipedia.set_lang("en")
create_dataset("train", 10)
create_dataset("test", 10)
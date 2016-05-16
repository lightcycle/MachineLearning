import wikipedia
import urllib
import cStringIO
import os
import traceback
import logging
import tempfile
import random
from PIL import Image

imagecount = 100
training_output_dir = "training"
mincolors = 1024
rotations = {0: '1000', 90: '0100', 180: '0010', 270: '0001'}

if not os.path.exists(training_output_dir):
    os.makedirs(training_output_dir)

random.seed()

wikipedia.set_lang("en")

while imagecount > 0:
    try:
        page = wikipedia.page(wikipedia.random())
        imageurls = [url for url in page.images if url.endswith(".png") or url.endswith(".jpg")]
        for imageurl in imageurls:
            file = cStringIO.StringIO(urllib.urlopen(imageurl).read())
            image = Image.open(file)
            width, height = image.size
            has_many_colors = image.getcolors(mincolors) is None
            if image.mode == "RGB" and width > 400 and height > 400 and has_many_colors:
                rotation = random.randint(0, len(rotations) - 1)
                tf = tempfile.NamedTemporaryFile(suffix = "-" + rotations.values()[rotation] + ".jpg", prefix = "", dir = training_output_dir, delete = False)
                image.thumbnail((400,400))
                new_image = Image.new("RGB", (400,400))
                new_image.paste(image, ((new_image.size[0] - image.size[0]) / 2, (new_image.size[1] - image.size[1]) / 2))
                new_image.rotate(rotations.keys()[rotation]).save(tf.name)
                imagecount -= 1
    except Exception as e:
        logging.error(traceback.format_exc())
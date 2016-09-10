import StringIO
from random import randint
from PIL import Image


class ExampleProducer:

    def __init__(self, width, height, min_colors, image_data_generator):
        self.width = width
        self.height = height
        self.min_colors = min_colors
        self.image_data_generator = image_data_generator
        self.labels = {0: (1, 0, 0, 0), 90: (0, 1, 0, 0), 180: (0, 0, 1, 0), 270: (0, 0, 0, 1)};

    def __usable_image(self, image):
        return image.mode == "RGB" and image.size >= (self.width, self.height) and image.getcolors(self.min_colors) is None

    @staticmethod
    def __center_crop_bounds(image):
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

    def example_generator(self):
        for image_data in self.image_data_generator:
            image = Image.open(StringIO.StringIO(image_data))
            if not self.__usable_image(image):
                continue
            angle = randint(0, 3) * 90
            image_data = StringIO.StringIO()
            image\
                .crop(self.__center_crop_bounds(image))\
                .resize((self.width, self.height))\
                .rotate(angle)\
                .save(image_data, 'JPEG')
            yield image_data.getvalue(), self.labels[angle]
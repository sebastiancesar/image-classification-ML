import base64
from PIL import Image
from StringIO import StringIO
import align.align_one as aligner
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input


class ImageProcessor:

    def __init__(self, crop_face):
        self.crop_face = crop_face
        self.aligner = aligner.Aligner()

    def process_base64(self, base64str):
        if len(base64str) == 0:
            raise Exception('Empty image string')
        image_data = base64.b64decode(base64str)
        ima = Image.open(StringIO(image_data))
        img_array = np.array(ima)
        if self.crop_face:
            img_array = self.aligner.process_one(img_array)
        return img_array.astype('float32')

    def get_img_array_from_base64(self, base64str):
        image_array = self.process_base64(base64str)
        image_array = np.expand_dims(image_array, 0)
        image_array = preprocess_input(image_array)
        return image_array

    def get_img_array_from_base64_batch(self, base64str):
        image_array = self.process_base64(base64str)
        image_array = preprocess_input(image_array)
        return image_array


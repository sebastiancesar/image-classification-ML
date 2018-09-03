import base64
from PIL import Image
from StringIO import StringIO
import align.align_one as aligner

CROP_FACES = True

class ImageProccesor:

    def __init__(self, crop_face=CROP_FACES):
        self.crop_face = crop_face
        self.aligner = aligner.Aligner()

    def process_base64(self, base64str):
        if len(base64str) == 0:
            raise Exception('Empty image string')
        image_data = base64.b64decode(base64str)
        ima = Image.open(StringIO(image_data))
        # ima = ImageOps.fit(ima, (160, 160))
        # ima.save('./pepe.jpg')
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

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def crop(self, image, random_crop, image_size):
        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            if random_crop:
                diff = sz1 - sz2
                (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            else:
                (h, v) = (0, 0)
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        return image

    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image

    def load_data(self, img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        if img.ndim == 2:
            img = self.to_rgb(img)
        if do_prewhiten:
            img = self.prewhiten(img)
        img = self.crop(img, do_random_crop, image_size)
        img = self.flip(img, do_random_flip)
        # images = np.expand_dims(img, 0)
        return img

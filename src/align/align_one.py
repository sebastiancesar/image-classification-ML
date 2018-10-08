from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc

import tensorflow as tf
import numpy as np
import base64
from align import detect_face
from StringIO import StringIO
from PIL import Image


IMAGE_SIZE = 224
MARGIN = 44
MULTIPLE_FACES = False
GPU_MEMORY_FRACTION = 1.0
IMAGE_SAMPLE = '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'

MIN_SIZE = 20  # minimum size of face
THRESHOLD = [0.6, 0.7, 0.7]  # three steps's threshold
FACTOR = 0.709  # scale factor


class Aligner:

    def __init__(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

    def process_one(self, img):
        # print('aligning > start ', str(datetime.datetime.now()))
        img = img[:, :, 0:3]

        bounding_boxes, _ = detect_face.detect_face(img, MIN_SIZE, self.pnet, self.rnet, self.onet,
            THRESHOLD, FACTOR)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces <= 0:
            print('**-*-*-*-* No face detected *-*-*----*')
            return img
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces == 1:
                det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - MARGIN / 2, 0)
                    bb[1] = np.maximum(det[1] - MARGIN / 2, 0)
                    bb[2] = np.minimum(det[2] + MARGIN / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + MARGIN / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear')

                # print('aligning > finish', str(datetime.datetime.now()))
                # imagen = Image.fromarray(scaled)
                # imagen.save('./tito.jpg')
                return scaled
            elif nrof_faces > 1:
                if MULTIPLE_FACES:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])

    def process_base64(base64str):
        image_data = base64.b64decode(base64str)
        ima = Image.open(StringIO(image_data))
        ima.save('./pepe.jpg')
        return np.array(ima).astype('float32')


# aligner = Aligner()
# img_array = aligner.process_one(aligner.process_base64(IMAGE_SAMPLE))
# imagen = Image.fromarray(img_array)
# imagen.save('./tito.jpg')

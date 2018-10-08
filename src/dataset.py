import tensorflow as tf

NUM_CLASS = 4


class Dataset:

    def __init__(self):
        self.xs = None
        self.ys = None
        self.class_samples = {}        

    def add_sample(self, activation, class_id):
        label_encoded = tf.one_hot(indices=[int(class_id)], depth=NUM_CLASS)

        if self.xs is None:
            self.xs = activation
        else:
            self.xs = tf.concat([self.xs, activation], 0)

        if self.ys is None:
            self.ys = label_encoded
        else:
            self.ys = tf.concat([self.ys, label_encoded], 0)

    def add_sample_img(self, class_id, img):
        if class_id not in self.class_samples:
            self.class_samples[class_id] = img
            return {'class_id': class_id, 'image': img}
        return False
    
    def get_samples_img(self):
        return self.class_samples

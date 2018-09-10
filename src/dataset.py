import tensorflow as tf

NUM_CLASS = 4


class Dataset:

    def __init__(self):
        self.xs = None
        self.ys = None

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

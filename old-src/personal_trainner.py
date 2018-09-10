import tensorflow as tf
import datetime
import numpy as np


STEPS_PER_EPOCH = 5
EPOCHS = 20
NUM_CLASS = 4
BATCH_SIZE = None


class PersonalTrainer:

    def __init__(self, predictor, image_processor, model_name):
        self.model_name = model_name
        self.image_processor = image_processor
        self.predictor = predictor
        self.xs = None
        self.ys = None

    def train(self, xs, ys, custom_model):
        if self.model_name == 'mobilenet':
            return self.train_mobilenet(xs, ys, custom_model)
        elif self.model_name == 'facenet':
            return self.train_mobilenet(xs, ys, custom_model)
            # return self.train_facenet(xs, ys, custom_model)

    def train_facenet (self, xs, ys, custom_model):
        xss = np.array(xs)
        history = custom_model.fit(xss, ys)
        print(history)


    def train_mobilenet (self, xs, ys, custom_model):
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))

            def on_train_end(self, logs={}):
                print('train ended', logs)

        loss_history = LossHistory()
        model_history = custom_model.fit(
            x=xs,
            y=ys,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            steps_per_epoch=STEPS_PER_EPOCH,  # Required if the input is a tensor data.
            callbacks=[loss_history])

        print(loss_history.losses)
        return model_history

    def process_images_batch(self, images_batch):
        xss = self.predictor.predict_mobilenet_batch(images_batch)
        if self.xs is None:
            self.xs = xss
        else:
            self.xs = tf.concat([self.xs, xss], 0)

    def process_labels_batch(self, class_id, batch_size):
        label_encoded = tf.one_hot(indices=[int(class_id)], depth=NUM_CLASS)
        yss = label_encoded
        for i in range(1, batch_size):
            yss = tf.concat([yss, label_encoded], 0)

        if self.ys is None:
            self.ys = yss
        else:
            self.ys = tf.concat([self.ys, yss], 0)

    def train_many_samples(self, training_samples, custom_model):
        print('starting to train ALL classes', str(datetime.datetime.now()))
        for class_id, samples in training_samples.iteritems():
            xss = list()
            print('starting to capture from mobilenet the class ', class_id)
            print('start to capture from mobilenet samples', str(datetime.datetime.now()))

            for img_str in samples:
                img_array = self.image_processor.get_img_array_from_base64_batch(img_str)
                xss.append(img_array)

            xss = np.array(xss)
            self.process_images_batch(xss)
            self.process_labels_batch(class_id, xss.shape[0])
            # print('finish training samples > mobilenet phase ', str(datetime.datetime.now()))
            # for element in xss:
            #     print('prediction for class ' + class_id + ' ', np.argmax(element))

        print('finish to cature ALL classes with mobilenet', str(datetime.datetime.now()))

        print('starting to training custom model', str(datetime.datetime.now()))
        history = self.train(self.xs, self.ys, custom_model)
        self.xs, self.ys = None, None
        print('finish training custom model', str(datetime.datetime.now()))
        return history
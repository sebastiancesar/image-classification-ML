import tensorflow as tf
import base64, datetime
from PIL import Image, ImageOps
import numpy as np
from StringIO import StringIO
import grpc
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from tensorflow.keras.applications.mobilenet import preprocess_input
import grpc._cython.cygrpc as cy


NUM_CLASS = 4
DENSE_UNITS = 100
BATCH_SIZE = 40
LEARNING_RATE = 0.0001
EPOCHS = 20
STEPS_PER_EPOCH = 1
INPUT_SHAPE_LAYER_SIZE = 256
ALPHA = 0.25


class Modeler:

    def __init__(self, load_mobilenet=False):
        if load_mobilenet:
            self.truncated_mobilenet = self.load_truncated_mobilenet()
        self.custom_model = self.load_custom()

    def export_model(self):
        tf.keras.backend.set_learning_phase(0)

        with tf.keras.backend.get_session() as sess:
            prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
                    {'image': self.truncated_mobilenet.input},
                    {'prediction': self.truncated_mobilenet.output})

            builder = tf.saved_model.builder.SavedModelBuilder('./exported_model')
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                }
            )
        builder.save()

    def load_custom(self):
        layer_flatten = tf.keras.layers.Flatten(input_shape=[7, 7, INPUT_SHAPE_LAYER_SIZE])
        layer_dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS,
                                              activation='relu',
                                              kernel_initializer='VarianceScaling',
                                              use_bias=True)
        layer_dense_2 = tf.keras.layers.Dense(units=NUM_CLASS,
                                              activation='softmax',
                                              kernel_initializer='VarianceScaling',
                                              use_bias=False)
        sequential = tf.keras.Sequential()
        sequential.add(layer_flatten)
        sequential.add(layer_dense_1)
        sequential.add(layer_dense_2)

        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        sequential.compile(optimizer=optimizer, loss='categorical_crossentropy')

        return sequential


    def load_truncated_mobilenet(self):

        mobilenet_model = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3),
            alpha=ALPHA,
            depth_multiplier=1,
            dropout=0.001,
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            classes=1000)

        layers = mobilenet_model.get_layer('conv_pw_13_relu')
        model = tf.keras.Model(inputs=mobilenet_model.input, outputs=layers.output)
        return model


class ImageProccesor:

    def process_base64(self, base64str):
        image_data = base64.b64decode(base64str)
        ima = Image.open(StringIO(image_data))
        # ima = ImageOps.fit(ima, (224, 224))
        ima.save('./pepe.jpg')
        return np.array(ima).astype('float32')

    def get_img_array_from_base64(self, base64str):
        image_array = self.process_base64(base64str)
        image_array = np.expand_dims(image_array, 0)
        image_array = preprocess_input(image_array)
        return image_array

    def get_img_array_from_base64_batch(self, base64str):
        image_array = self.process_base64(base64str)
        image_array = preprocess_input(image_array)
        return image_array

class PersonalTrainer:

    def __init__(self, predictor, custom_model, image_processor):
        self.image_processor = image_processor
        self.predictor = predictor
        self.custom_model = custom_model

    def train(self, xs, ys, custom_model):
        class TellMeWhenFinished(tf.keras.callbacks.Callback):
            def on_train_end(self, logs={}):
                print('train ended', logs)

        model_history = custom_model.fit(
            x=xs,
            y=ys,
            batch_size=None,  # changed: If steps_per_epoch is set, the `batch_size` must be None.
            epochs=EPOCHS,
            shuffle=True,
            steps_per_epoch=STEPS_PER_EPOCH,  # Required if the input is a tensor data.
            callbacks=[TellMeWhenFinished()])

        return model_history

    def train_many_samples(self, training_samples, custom_model):
        print('starting to train ALL classes', str(datetime.datetime.now()))
        xs, ys = None, None
        for class_id, samples in training_samples.iteritems():
            yss, xss = list(), list()
            print('starting to capture from mobilenet the class ', class_id)
            print('start to capture from mobilenet samples', str(datetime.datetime.now()))
            for img_str in samples:
                img_array = self.image_processor.get_img_array_from_base64_batch(img_str)
                xss.append(img_array)

            xss = np.array(xss)
            xss = self.predictor.predict_mobilenet_batch(xss)
            if xs is None:
                xs = xss
            else:
                xs = tf.concat([xs, xss], 0)

            # print('finish training samples > mobilenet phase ', str(datetime.datetime.now()))
            # for element in xss:
            #     print('prediction for class ' + class_id + ' ', np.argmax(element))

            label_encoded = tf.one_hot(indices=[int(class_id)], depth=NUM_CLASS)
            yss = label_encoded
            for i in range(1, xss.shape[0]):
                yss = tf.concat([yss, label_encoded], 0)
            if ys is None:
                ys = yss
            else:
                ys = tf.concat([ys, yss], 0)

        print('finish to cature ALL classes with mobilenet', str(datetime.datetime.now()))

        print('starting to training custom model', str(datetime.datetime.now()))
        self.train(xs, ys, custom_model)
        print('finish training custom model', str(datetime.datetime.now()))


class Clasificados:

    def __init__(self, local=False):
        self.default_graph = tf.get_default_graph()
        self.image_processor = ImageProccesor()
        self.modelero = Modeler(load_mobilenet=local)
        if local:
            self.predictor = LocalMobilenetPredictor(self.modelero)
        else:
            self.predictor = MobilenetPredictor(self.modelero)
        self.personal_trainer = PersonalTrainer(self.predictor, self.modelero.custom_model, self.image_processor)

    def train_samples(self, samples):
        with self.default_graph.as_default():
            custom_model = self.modelero.load_custom()
            self.personal_trainer.train_many_samples(samples, custom_model)

    def predict(self, img_str):
        with self.default_graph.as_default():

            #print('about to predict one image ', str(datetime.datetime.now()))
            img_array = self.image_processor.get_img_array_from_base64(img_str)
            activation = self.predictor.predict_mobilenet(img_array)
            print(np.argmax(activation))
            prediction = self.predictor.predict_custom(activation)
            resolved_class = np.argmax(prediction)
            confidence = prediction[0][resolved_class]
            confidence = str(round(confidence, 2))
            result = {'class_id': resolved_class, 'confidence': confidence}
            #print('prediction end', str(datetime.datetime.now()))
            print(result)
            return result


class ModelPredictor(object):

    def __init__(self, modelero):
        self.custom_model = modelero.custom_model

    def predict_custom(self, img_array):
        return self.custom_model.predict(img_array)


class MobilenetPredictor(ModelPredictor):

    def __init__(self, modelero):
        super(MobilenetPredictor, self).__init__(modelero)
        self.channel_served = grpc.insecure_channel('localhost:9000',
                [(cy.ChannelArgKey.max_send_message_length, -1),
                 (cy.ChannelArgKey.max_receive_message_length, -1)])

    def predict_mobilenet_batch(self, images_batch):
        stub = prediction_service_pb2.PredictionServiceStub(self.channel_served)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mobilenet'
        request.inputs['image'].CopyFrom(tf.make_tensor_proto(images_batch))
        result = stub.Predict(request, 20.0)
        proto_result = result.outputs['prediction']
        return tf.make_ndarray(proto_result)

    def predict_mobilenet(self, image_array):
        stub = prediction_service_pb2.PredictionServiceStub(self.channel_served)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mobilenet'
        request.inputs['image'].CopyFrom(tf.make_tensor_proto(np.array(image_array)))
        result = stub.Predict(request, 5.0)
        proto_result = result.outputs['prediction']
        return tf.make_ndarray(proto_result)


class LocalMobilenetPredictor(ModelPredictor):

    def __init__(self, modelero):
        super(LocalMobilenetPredictor, self).__init__(modelero)
        self.truncated_mobilenet = modelero.truncated_mobilenet

    def predict_mobilenet(self, image_batched):
        prediction = self.truncated_mobilenet.predict(image_batched, steps=1, verbose=1)
        #print(np.argmax(prediction))
        return prediction

    def predict_mobilenet_batch(self, images_batch):
        return self.predict_mobilenet(images_batch)


import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.svm import SVC

INPUT_SHAPE_LAYER_SIZE = 256
CUSTOM_MODEL_INPUT_SHAPE = [7, 7, INPUT_SHAPE_LAYER_SIZE]
CUSTOM_MODEL_FACENET_INPUT_SHAPE = [128]
DENSE_UNITS = 100
NUM_CLASS = 4
LEARNING_RATE = 0.0001
ALPHA = 0.25



class Modeler:

    def __init__(self, load_mobilenet=False, model_name='mobilenet'):
        self.model_name = model_name
        if load_mobilenet:
            self.truncated_mobilenet = self.load_truncated_mobilenet()

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

    def export_model_tfjs(self):
        tfjs.converters.save_keras_model(self.truncated_mobilenet, './tfjs-model')

    def load_custom(self):
        if self.model_name == 'mobilenet':
            return self.custom_for_mobilenet()
        elif self.model_name == 'facenet':
            return self.custom_for_facenet()

    def custom_for_facenet(self):
        # model = SVC(kernel='linear', probability=True)
        # return model
        layer_flatten = tf.keras.layers.Flatten(input_shape=CUSTOM_MODEL_FACENET_INPUT_SHAPE)
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

    def custom_for_mobilenet(self):
        layer_flatten = tf.keras.layers.Flatten(input_shape=CUSTOM_MODEL_INPUT_SHAPE)
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
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            classes=1000)

        layers = mobilenet_model.get_layer('conv_pw_13_relu')
        model = tf.keras.Model(inputs=mobilenet_model.input, outputs=layers.output)
        return model

import tensorflow as tf
import tensorflowjs as tfjs


INPUT_SHAPE_LAYER_SIZE = 256
CUSTOM_MODEL_INPUT_SHAPE = [7, 7, INPUT_SHAPE_LAYER_SIZE]
CUSTOM_MODEL_FACENET_INPUT_SHAPE = [128]
DENSE_UNITS = 100
NUM_CLASS = 4
LEARNING_RATE = 0.0001
ALPHA = 0.25


class ModelManager:

    def load_custom(self):
        return self.custom_for_mobilenet()

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

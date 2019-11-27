import tensorflow as tf


STEPS_PER_EPOCH = 5
EPOCHS = 20
NUM_CLASS = 4
BATCH_SIZE = None


class PersonalTrainer:

    def train(self, xs, ys, custom_model):
        return self.train_mobilenet(xs, ys, custom_model)

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

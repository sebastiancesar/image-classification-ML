import tensorflow as tf
import numpy as np
import grpc
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2

import grpc._cython.cygrpc as cy
from modeler import Modeler
from personal_trainner import PersonalTrainer


NUM_CLASS = 4
MODEL_NAME = 'mobilenet'
CROP_FACES = True


class FacenetDataset:

    def __init__(self, model_name):
        self.xs = None
        self.ys = None
        self.model_name = model_name

    def add_sample(self, activation, class_id):

        if self.xs is None:
            self.xs = activation
        else:
            self.xs = np.concatenate((self.xs, activation), 0)

        if self.ys is None:
            self.ys = [class_id]
        else:
            self.ys.append(class_id)


class Clasificados:

    def __init__(self, local=False):
        self.default_graph = tf.get_default_graph()
        self.image_processor = ImageProccesor()
        self.modelero = Modeler(load_mobilenet=local, model_name=MODEL_NAME)
        self.model_name = MODEL_NAME
        self.dataset = Dataset(model_name=MODEL_NAME)
        self.custom_model = self.modelero.load_custom()

        if local:
            self.predictor = LocalMobilenetPredictor(self.modelero)
        elif self.model_name == 'facenet':
            self.predictor = FacenetPredictor(self.modelero, self.image_processor)
        else:
            self.predictor = MobilenetPredictor(self.modelero)

        self.personal_trainer = PersonalTrainer(self.predictor, self.image_processor, model_name=MODEL_NAME)

    def reset(self):
        self.dataset = Dataset(model_name=MODEL_NAME)
        print('reseted')

    def clean_data_holder(self):
        self.dataset = Dataset(model_name=MODEL_NAME)

    def train(self):
        self.custom_model = self.modelero.load_custom()
        history = self.personal_trainer.train(self.dataset.xs, self.dataset.ys, self.custom_model)
        self.clean_data_holder()

    def add_sample(self, img_str, class_id):
        img_array = self.image_processor.get_img_array_from_base64(img_str)
        activation = self.predictor.predict_mobilenet(img_array)
        self.dataset.add_sample(activation, class_id)
        print('sample added for class ', class_id)

    def train_samples(self, samples):
        custom_model = self.modelero.load_custom()
        return self.personal_trainer.train_many_samples(samples, custom_model)

    def predict(self, img_str):
        # validate there is a custom_model
        img_array = self.image_processor.get_img_array_from_base64(img_str)
        activation = self.predictor.predict_mobilenet(img_array)
        # print(np.argmax(activation))
        prediction = self.predictor.predict_custom(activation, self.custom_model)
        if self.model_name == 'facenet':
            best_class_indices = np.argmax(prediction, axis=1)
            best_class_probabilities = prediction[np.arange(len(best_class_indices)), best_class_indices]
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, [best_class_indices[i]], best_class_probabilities[i]))

            confidence = str(round(best_class_probabilities[0], 2))
            result = {'class_id': best_class_indices[0], 'confidence': confidence }
        else:
            resolved_class = np.argmax(prediction)
            confidence = prediction[0][resolved_class]
            confidence = str(round(confidence, 2))
            result = {'class_id': resolved_class, 'confidence': confidence}

        print(result)
        return result


class ModelPredictor(object):

    def predict_custom(self, img_array, model):
        if MODEL_NAME == 'facenet':
            return model.predict_proba(img_array)
        else:
            return model.predict(x=img_array, batch_size=1, verbose=1)


class MobilenetPredictor(ModelPredictor):

    def __init__(self, modelero):
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
        self.truncated_mobilenet = modelero.truncated_mobilenet

    def predict_mobilenet(self, image_batched):
        prediction = self.truncated_mobilenet.predict(x=image_batched, batch_size=1, verbose=1)
        #print(np.argmax(prediction))
        return prediction

    def predict_mobilenet_batch(self, images_batch):
        return self.predict_mobilenet(images_batch)


class FacenetPredictor(ModelPredictor):

    def __init__(self, modelero, image_processor):
        self.channel_served = grpc.insecure_channel('localhost:9000',
                [(cy.ChannelArgKey.max_send_message_length, -1),
                (cy.ChannelArgKey.max_receive_message_length, -1)])

        self.image_processor = image_processor

    def predict_mobilenet(self, image):
        images = self.image_processor.load_data(image, False, False, 160)
        stub = prediction_service_pb2.PredictionServiceStub(self.channel_served)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'facenet'
        request.model_spec.signature_name = 'calculate_embeddings'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(images))
        request.inputs['phase'].CopyFrom(tf.make_tensor_proto(False))
        result = stub.Predict(request, 20.0)
        proto_result = result.outputs['embeddings']
        return tf.make_ndarray(proto_result)

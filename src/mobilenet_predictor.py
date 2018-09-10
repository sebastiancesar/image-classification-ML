import tensorflow as tf
import grpc
import grpc._cython.cygrpc as cy
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
import numpy as np


MOBILENET_SERVING_ADDRESS = 'localhost:9000'  # '142.93.111.69:8500'


class MobilenetPredictor:

    def __init__(self):
        self.channel_served = grpc.insecure_channel(MOBILENET_SERVING_ADDRESS,
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

    def predict_mobilenet_json(self, image_array):
        image_array


    def predict_custom(self, img_array, model):
        return model.predict(x=img_array, batch_size=1, verbose=1)

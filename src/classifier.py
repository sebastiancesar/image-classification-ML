from image_processor import ImageProcessor
from model_manager import ModelManager
from dataset import Dataset
from mobilenet_predictor import MobilenetPredictor
from personal_trainner import PersonalTrainer
import numpy as np
import logger_config

logger = logger_config.get_logger(__name__)

CROP_FACES = True


class CaripelaClassifier:

    def __init__(self):
        self.image_processor = ImageProcessor(crop_face=CROP_FACES)
        self.model_manager = ModelManager()
        # initialize volatile variables
        self.prediction_model = MobilenetPredictor()
        self.personal_trainer = PersonalTrainer()
        self.sessions = {}

    def create_session(self, session_id):
        self.sessions[session_id] = {
            'dataset': Dataset(),
            'custom_model': self.model_manager.load_custom()
        }
        logger.debug('create_session > session created with id %s', session_id)

    def reset(self, session_id):
        self.clean_data_holder(session_id)
        self.sessions[session_id]['custom_model'] = self.model_manager.load_custom()
        print('reseted')

    def clean_data_holder(self, session_id):
        try:
            self.sessions[session_id]['dataset'] = Dataset()
        except KeyError:
            logger.debug('session_id %s not exits', session_id)

    def train(self, session_id):
        custom_model = self.sessions[session_id]['custom_model']  # model_manager.load_custom()
        dataset = self.sessions[session_id]['dataset']
        history = self.personal_trainer.train(dataset.xs, dataset.ys, custom_model)
        self.clean_data_holder(session_id)
        logger.debug('traing > training completed for session %s', session_id)
        return history

    def add_sample(self, session_id, img_str, class_id):
        img_array, img_face = self.image_processor.get_img_array_from_base64(img_str)
        activation = self.prediction_model.predict_mobilenet(img_array)
        dataset = self.sessions[session_id]['dataset']
        dataset.add_sample(activation, class_id)
        logger.debug('sample added for class %s for session_id %s', class_id, session_id)
        #return self.add_sample_img(session_id, class_id, img_face)


    def predict(self, session_id, img_str):
        # validate there is a custom_model
        custom_model = self.sessions[session_id]['custom_model']
        img_array, img_face = self.image_processor.get_img_array_from_base64(img_str)
        activation = self.prediction_model.predict_mobilenet(img_array)
        prediction = self.prediction_model.predict_custom(activation, custom_model)
        resolved_class = np.argmax(prediction)
        confidence = prediction[0][resolved_class]
        confidence = str(round(confidence, 2))
        result = {'class_id': resolved_class, 'confidence': confidence}
        logger.debug(result)
        return result

    def add_sample_img(self, session_id, class_id, img_face_array):
        dataset = self.sessions[session_id]['dataset']
        return dataset.add_sample_img(class_id, self.image_processor.array_to_base64(img_face_array))

    def get_classes_img(self, session_id):
        dataset = self.sessions[session_id]['dataset']
        return dataset.get_samples_img()


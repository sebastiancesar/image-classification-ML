from flask import Flask, request

from flask_socketio import SocketIO, emit
from classifier import CaripelaClassifier
from flask_cors import CORS
import sys
import logger_config


logger = logger_config.get_logger('web-server')


def log_uncaught(exctype, value, tb):
    logger.error('Type: ', exctype)
    logger.error('Value:', value)
    logger.error('Traceback:', tb)


#sys.excepthook = logger
# initialize Flask
app = Flask(__name__)
cors = CORS(app)

socketio = SocketIO(app)
clasificados = CaripelaClassifier()


""" Handle connections """

@socketio.on('connect')
def on_connect():
    print('new client connected')


""" PARA EL BACKEND DE KERAS """


@socketio.on('backend_new_session')
def backend_new_session(data):
    clasificados.create_session(session_id=data['sessionId'])


@socketio.on('backend_reset')
def backend_reset(data):
    clasificados.reset(session_id=data['sessionId'])


@socketio.on('backend_add_sample')
def backend_add_sample(data):
    clasificados.add_sample(session_id=data['sessionId'],
                            img_str=data['sample'],
                            class_id=data['class_id'])


@socketio.on('backend_train')
def backend_train(data):
    clasificados.train(session_id=data['sessionId'])
    emit('training_completed')


@socketio.on('backend_predict')
def backend_predict(data):
    prediction = clasificados.predict(session_id=data['sessionId'], img_str=data['sample'])
    emit('server_predicted', prediction)
    return prediction


if __name__ == '__main__':
    socketio.run(app, host='192.168.0.155',
                 debug=True,
                 use_reloader=False,
                 keyfile='./src/certificates/server.key',  # key.pem',
                 certfile='./src/certificates/server.crt')  #cert.pem')


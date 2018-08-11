from flask import Flask
from flask_socketio import SocketIO, emit
from main import Clasificados

# initialize Flask
app = Flask(__name__)
socketio = SocketIO(app)
rooms = {}  # dict to track active rooms
print ('**********************ABOUT TO CREATE A PREDICTOR********************************')
clasificados = Clasificados(local=True)


@socketio.on('response_connect')
def on_connected(data):
    print(data['msg'])


@socketio.on('train')
def train(samples):
    clasificados.train_samples(samples)
    emit('train_completed', {'status': 'ok'})


@socketio.on('predict')
def on_predict(data):
    prediction = clasificados.predict(data['imageData'])
    return prediction



if __name__ == '__main__':
    socketio.run(app, host='192.168.0.156', debug=True, use_reloader=False, keyfile='key.pem', certfile='cert.pem')


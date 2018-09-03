from flask import Flask
from flask_socketio import SocketIO, emit, join_room
from main import Clasificados
from flask_cors import CORS


# initialize Flask
app = Flask(__name__)
cors = CORS(app)

socketio = SocketIO(app)
clasificados = Clasificados(local=False)



""" Handle connections """


@socketio.on('response_connect')
def on_connected(data):
    print(data['msg'])


@socketio.on('create')
def on_connection(data):
    room_id = data['roomId']
    join_room(room_id)
    emit('joined_room', {'roomId': room_id})
    clasificados.reset()


""" Handle request from BROWSER"""


@socketio.on('browser_test_start_record')
def browser_test_start_record():
    emit('browser_test_start_record', {}, room='serverRoom')


@socketio.on('browser_test_stop_record')
def browser_test_stop_record():
    emit('browser_test_stop_record', {}, room='serverRoom')


@socketio.on('browser_sample_added')
def browser_sample_added(data):
    print(data)
    emit('sample_added', room='clientRoom')


@socketio.on('browser_train')
def browser_train():
    print('server_train forwarded')
    emit('browser_train', room='serverRoom')


@socketio.on('browser_add_sample')
def browser_add_sample(data):
    emit('browser_add_sample', data, room='serverRoom')
    print ('add_sample forwarded')


@socketio.on('browser_predicted')
def browser_predicted(data):
    print('browser server predicted', data)
    emit('server_predicted', data, room='clientRoom')


@socketio.on('browser_predict')
def browser_predict(data):
    emit('browser_predict', data, room='serverRoom')


""" PARA EL BACKEND DE KERAS """

@socketio.on('backend_reset')
def backend_reset():
    clasificados.reset()

@socketio.on('backend_add_sample')
def backend_add_sample(data):
    clasificados.add_sample(img_str=data['sample'], class_id=data['class_id'])


@socketio.on('backend_train')
def backend_train():
    clasificados.train()
    emit('training_completed')


# DEPRECATED - Use add_sample instead.
@socketio.on('backend_train_samples')
def backend_train_samples(samples):
    clasificados.train_samples(samples)
    emit('train_completed', {'status': 'ok'})


@socketio.on('backend_predict')
def backend_predict(data):
    prediction = clasificados.predict(data['sample'])
    emit('server_predicted', prediction)
    return prediction


if __name__ == '__main__':
    socketio.run(app, host='192.168.0.155', debug=True, use_reloader=False, keyfile='key.pem', certfile='cert.pem')


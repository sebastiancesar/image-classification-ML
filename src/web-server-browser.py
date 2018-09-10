from flask_socketio import SocketIO, emit


# socketio = SocketIO(app)

class BrowserHandler:

    def __init__(self, socketio):
        self.socketio = socketio

    """ Handle request from BROWSER"""

    @socketio.on('browser_test_start_record')
    def browser_test_start_record(self):
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

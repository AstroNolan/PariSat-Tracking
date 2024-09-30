from gevent import monkey
monkey.patch_all()

from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

if __name__ == '__main__':
    server = WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    socketio.run(app, server=server)
import random
import logging
import socket
import webview
import threading
import uvicorn
from main import app

logging.basicConfig(level=logging.INFO)


def get_random_port() -> int:
    while True:
        port = random.randint(1023, 65535)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))
            except OSError:
                logging.warning('Port %s is in use' % port)
                continue
            else:
                return port


server_port = get_random_port()
logging.info(f'Starting server on port {server_port}')
server_thread = threading.Thread(
    target=uvicorn.run, args=(app,), kwargs={'port': server_port})
server_thread.start()
window = webview.create_window(
    'RedbookLang IDE', f'http://localhost:{server_port}',
    width=1200, height=800,
    text_select=True
)
webview.start()

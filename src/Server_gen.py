import socket
import threading
import os
import json
import base64
from src.image_gen import ImageGen


class server:

    def __init__(self):

        print("---Server Initializing---")
        # create a socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # get local machine name
        self.host = socket.gethostname()
        # bind the socket to a public host, and a port
        self.server_socket.bind((self.host, 9999))
        # set the server to listen for incoming requests
        self.server_socket.listen(5)

        self.image_gen = ImageGen()

    def handle_request(self, client_socket, addr):

        data = client_socket.recv(1024)

        # parse the JSON data
        json_data = json.loads(data.decode())

        # extract the image and text data from the JSON data
        task_data = json_data['task']
        text_data = json_data['text']
        image_data = base64.b64decode(json_data['image'])

        print("type image_data", type(image_data))

        if task_data == 0: result = self.image_gen.img2img(image_data)
        elif task_data == 1: result = self.image_gen.img2img_clip(image_data)
        else: result= self.image_gen.text2img(image_data)

        print("type result: ", type(result))

        # close the client socket
        client_socket.close()

    def start(self):
        print("---Server Start!!---")
        while True:
            # wait for a client to connect
            client_socket, addr = self.server_socket.accept()

            # create a new thread to handle the request
            t = threading.Thread(target=self.handle_request, args=(client_socket, addr))
            t.start()
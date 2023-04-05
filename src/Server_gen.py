import socket
import json
import base64
import threading
from io import BytesIO
from PIL import Image
from src.image_gen import ImageGen

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def process_request(data):
    json_data = json.loads(data)
    image = base64_to_image(json_data['image'])

    print("Task: ", json_data['task'])
    print("Image size: ", image.size)

def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

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

    def handle_request(self, conn, addr):
        with conn:
            print(f"Connected by {addr}")

            # Receive the data length first
            data_length = int.from_bytes(conn.recv(4), 'big')

            # Receive the data in chunks
            data = recv_all(conn, data_length)
            if not data:
                return

            json_data_decode = data.decode('utf-8')
            json_data = json.loads(json_data_decode)
            # extract the image and text data from the JSON data
            task_data = json_data['task']
            image = base64_to_image(json_data['image'])

            if task_data == 0:
                result = self.image_gen.img2img(image)
                result_image = result[0]
                result_image.save("./output/1.jpg")
            elif task_data == 1:
                result = self.image_gen.img2img_clip(image)
                result_clip_text = result[0]
                result_image = result[1]
                result_image.save("./output/1.jpg")
            else:
                result = self.image_gen.text2img(image)

            print("result type", type(result))
            print()
            result[1].save()

            response = "Image received and processed."
            conn.sendall(response.encode('utf-8'))


    def start(self):
        print("---Server Start!!---")
        print(self.server_socket.getsockname()[0])
        while True:
            # wait for a client to connect
            conn, addr = self.server_socket.accept()

            # create a new thread to handle the request
            t = threading.Thread(target=self.handle_request, args=(conn, addr))
            t.start()







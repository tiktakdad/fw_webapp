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
        self.port = 9999
        # create a socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # get local machine name
        #self.host = socket.gethostname()
        # bind the socket to a public host, and a port
        #self.server_socket.bind((self.host, self.port))
        self.server_socket.bind(('', self.port))
        # set the server to listen for incoming requests
        self.server_socket.listen(1)

        self.image_gen = ImageGen()

    def handle_request(self, conn, addr):
        with conn:
            print('1')
            print(f"Connected by {addr}")

            # Receive the data length first
            data_length = int.from_bytes(conn.recv(4), 'little')
            print('2')
            print(data_length)
            # Receive the data in chunks
            data = recv_all(conn, data_length)
            if not data:
                return
            print('3')
            json_data_decode = data.decode('utf-8')
            json_data = json.loads(json_data_decode)
            # extract the image and text data from the JSON data
            task_data = json_data['task']
            image = base64_to_image(json_data['image'])
            image.save("./output/test.jpg")
            print(task_data)
            #print(json_data['image'])


            if task_data == 0:
                result = self.image_gen.img2img(image)
                result_image = result[0]
                result_image.save("./output/1.jpg")
            elif task_data == 1:
                result = self.image_gen.img2img_clip(image)
                result_clip_text = result[0]
                result_image = result[1]
                result_image.save("./output/2.jpg")
            else:
                ocr_trans, prompt, output_image = self.image_gen.text2img(image)
                print(ocr_trans)
                print(prompt)
                output_image.save("./output/3.jpg")
                image_path = './resource/diary/diary_sample.png'
                encoded_image = self.image_to_base64("./output/3.jpg")
                print('encoded_image: ',encoded_image[0:5], ' len: ', len(encoded_image))
                #encoded_image = base64.b64encode(output_image).decode('utf-8')
                encoded_image_length = len(encoded_image)
            

            
            conn.sendall(encoded_image_length.to_bytes(4, 'little'))
            conn.sendall(encoded_image.encode('utf-8'))


    def image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def start(self):
        print("---Server Start!!---")
        print(self.server_socket.getsockname()[0],' ', self.port)
        while True:
            # wait for a client to connect
            conn, addr = self.server_socket.accept()

            # create a new thread to handle the request
            t = threading.Thread(target=self.handle_request, args=(conn, addr))
            t.start()







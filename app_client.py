import cv2
import requests
import json
import os, base64
import socket

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_image(data, server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_ip, server_port))

        # Send the data length first
        data_length = len(data)
        sock.sendall(data_length.to_bytes(4, 'big'))

        # Send data to the server
        sock.sendall(data.encode('utf-8'))

        # Receive response from the server
        response = sock.recv(1024)

    return response

if __name__ == '__main__':
    server_ip = '127.0.1.1'
    server_port = 9999

    task_num = 1
    image_path = './resource/coloring/sample (1).png'

    # Convert image to base64
    encoded_image = image_to_base64(image_path)

    # Create JSON object with the base64-encoded image
    data = json.dumps({'task': task_num, 'image': encoded_image})

    #print("json:", data)

    # Send image
    image_response = send_image(data, server_ip, server_port)
    print('Image response:', image_response)


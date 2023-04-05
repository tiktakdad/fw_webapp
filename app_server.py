import gradio as gr
from PIL import Image
from src.image_gen import ImageGen
from src.Server_gen import server


if __name__ == "__main__":
    Server = server()
    Server.start()

import gradio as gr
from PIL import Image
from src.image_gen import ImageGen
import json


def read_img2img_json(path):
    # './resource/coloring/coloring.json'
    with open(path, 'r') as file:
        img2img_json_data = json.load(file)
    return img2img_json_data

def coloring_book_json2examplesList(json_data):
    json_keys = json_data.keys()
    list_filepaths = []
    for filename in json_keys:
        list_filepaths.append([json_data[filename]['path'], json_data[filename]['body'],filename])
    return list_filepaths

def free_sketch_json2examplesList(json_data):
    json_keys = json_data.keys()
    list_filepaths = []
    for filename in json_keys:
        list_filepaths.append(json_data[filename]['path'])
    return list_filepaths

def run_app():
    # TODO: 각 탭 별 코드 클래스 및 py 파일로 분할 하여 개발
    # for test

    img2img_json_path = './resource/coloring/coloring.json'
    img2img_json_data = read_img2img_json(img2img_json_path)

    img2img_clip_json_path = './resource/free_sketch/free_sketch.json'
    img2img_clip_json_data = read_img2img_json(img2img_clip_json_path)

    # create image generation model
    img_gen = ImageGen()

    # create gradio
    md = "🐳 Flying Whales"
    app1 = gr.Interface(fn=img_gen.img2img, 
                inputs=[gr.Image(type="pil"), gr.inputs.Textbox(label="sketch label"), gr.inputs.Textbox(label="sample index")],
                outputs=gr.Image(type="pil").style(width=512, height=512),
                examples=coloring_book_json2examplesList(img2img_json_data)
                #examples=["resource/coloring/sample (1).png", "resource/coloring/sample (2).png"]
                )
    app2 = gr.Interface(fn=img_gen.img2img_clip, 
                inputs=gr.Image(type="pil"),
                outputs=["text", gr.Image(type="pil").style(width=512, height=512)],
                examples=free_sketch_json2examplesList(img2img_clip_json_data))
    app3 = gr.Interface(fn=img_gen.text2img, 
                        inputs=gr.Image(type="pil"), 
                        outputs=["text","text", gr.Image(type="pil").style(width=632, height=408)], 
                        examples=["resource/diary/sample/diary_sample (5).jpg", "resource/diary/sample/diary_sample (6).jpg"])
    demo = gr.TabbedInterface(title=md, interface_list=[app1, app2, app3], tab_names=["coloring book", "free drawing","diary"])
    #demo.launch(auth=("admin", "admin"), server_name='0.0.0.0')
    demo.launch(server_name='0.0.0.0')


if __name__ == "__main__":
    run_app()
    #init_image = Image.open("resource/sample (1).png").convert("RGB")
    #img2img(init_image)
    

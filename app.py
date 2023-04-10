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
                description="[색칠놀이]\n펜으로 컬러링북에 색칠을 한 상태라고 가정한 시연용 모드입니다.\n컬러링북 자체에는 정해진 프롬프트가 설정 되어있기 때문에, 직접 업로드 한 이미지에는 결과물이 좋지 않을 수 있습니다.",
                inputs=[gr.Image(type="pil"), gr.inputs.Textbox(label="sketch label"), gr.inputs.Textbox(label="sample index")],
                outputs=gr.Image(type="pil").style(width=512, height=512),
                examples=coloring_book_json2examplesList(img2img_json_data)
                #examples=["resource/coloring/sample (1).png", "resource/coloring/sample (2).png"]
                )
    app2 = gr.Interface(fn=img_gen.img2img_clip, 
                inputs=gr.Image(type="pil"),
                description="[자유그리기]\n펜으로 자유롭게 그림을 그린 상태라고 가정한 시연용 모드입니다.\n입력된 이미지를 기반으로 객체인식을 하여 이미지를 랜덤하게 생성 해줍니다.",
                #outputs=["text", gr.Image(type="pil").style(width=512, height=512)],
                outputs=[ gr.Textbox(label="image captioning from image "), gr.Image(type="pil")],
                examples=free_sketch_json2examplesList(img2img_clip_json_data))
    
    app3 = gr.Interface(fn=img_gen.text2img, 
                        inputs=gr.Image(type="pil"), 
                        #outputs=["text","text", gr.Image(type="pil").style(width=632, height=408)], 
                        description="[그림일기]\n펜으로 노트에 일기를 작성 하였다고 가정한 시연용 모드입니다.\n입력된 이미지에서 OCR을 통해 한글을 인식하고, ChatGPT를 통해 이미지생성에 적합한 Prompt를 생성합니다.\n 이를 기반으로 이미지를 랜덤하게 생성 해줍니다.",
                        outputs=[gr.Textbox(label="OCR and translation(Kor->Eng)"), gr.Textbox(label="ChatGPT prompt Maker "),  gr.Image(type="pil")], 
                        layout="vertical",
                        examples=["resource/diary/sample/diary_sample (5).jpg", "resource/diary/sample/diary_sample (6).jpg"])
    
    demo = gr.TabbedInterface(title=md, interface_list=[app1, app2, app3], tab_names=["coloring book", "free drawing","diary"])
    demo.launch(auth=("admin", "1004"), server_name='0.0.0.0')
    #demo.launch(server_name='0.0.0.0')


if __name__ == "__main__":
    run_app()
    #init_image = Image.open("resource/sample (1).png").convert("RGB")
    #img2img(init_image)
    

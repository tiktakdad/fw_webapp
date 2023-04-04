import gradio as gr
from PIL import Image
from src.image_gen import ImageGen



    
def run_app():
    # TODO: 각 탭 별 코드 클래스 및 py 파일로 분할 하여 개발
    # for test

    # create image generation model
    img_gen = ImageGen()

    # create gradio
    md = "🐳 Flying Whales"
    app1 = gr.Interface(fn=img_gen.img2img, 
                inputs=gr.Image(type="pil"),
                outputs=gr.Image(type="pil").style(width=256, height=384),
                examples=["resource/coloring/sample (1).png", "resource/coloring/sample (2).png"])
    app2 = gr.Interface(fn=img_gen.img2img_clip, 
                inputs=gr.Image(type="pil"),
                outputs=["text", gr.Image(type="pil").style(width=512, height=512)],
                examples=["resource/coloring/sample (11).png"])
    app3 = gr.Interface(fn=img_gen.text2img, 
                        inputs=gr.Image(type="pil"), 
                        outputs=["text", gr.Image(type="pil")], 
                        examples=["resource/diary/sample.jpg"])
    demo = gr.TabbedInterface(title=md, interface_list=[app1, app2, app3], tab_names=["coloring book", "free drawing","diary"])
    demo.launch()


if __name__ == "__main__":
    run_app()
    #init_image = Image.open("resource/sample (1).png").convert("RGB")
    #img2img(init_image)
    

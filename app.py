import gradio as gr
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

def ocr(split_imgs):
    # TODO: ocr from split_imgs to result
    result = '오늘은, 재밌었다.'
    return result

def text2img(img):
    input_rows = 5
    input_cols = 10
    start_point = (82,702)
    box = 63
    margin = 0
    split_imgs = []
    for rows in range(input_rows):
        for cols in range(input_cols):
            left = cols*box+start_point[0]+margin
            right = left+box
            top = rows*box+start_point[1]+margin
            bottom = top+box
            split_img = img.crop((left, top, right, bottom))
            split_img.save("output/{}_{}.jpg".format(rows, cols))
            split_imgs.append(split_img)
    
    #im1 = im.crop((left, top, right, bottom))
    return ocr(split_imgs)

def img2img(img):
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    #model_id_or_path = "model/dump"
    #pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config, use_karras_sigmas=True)
    pipe = pipe.to(device)

    #url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    #response = requests.get(url)
    #init_image = Image.open(BytesIO(img)).convert("RGB")
    init_image = img
    init_image = init_image.resize((512, 768))
    generator = torch.manual_seed(3306892559)

    prompt = "masterpiece, best quality, ultra-detailed, illustration,  1princess"
    negative_prompt = "worst quality, low quality, realistic, text, title, logo, signature, abs, muscular, rib, depth of field, bokeh, blurry, greyscale, monochrome"
    print("eval")
    images = pipe(prompt=prompt,negative_prompt=negative_prompt, image=init_image, generator=generator, num_inference_steps=40, guidance_scale=7).images
    '''
    upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]
    print(type(images[0]))
    # images[0].save("fantasy_landscape.png")
    '''
    return images[0]


    
def run_app():
    # TODO: 각 탭 별 코드 클래스 및 py 파일로 분할 하여 개발
    md = "🐳 Flying Whales"
    app1 = gr.Interface(fn=img2img, 
                inputs=gr.Image(type="pil"),
                outputs=gr.Image(type="pil").style(width=256, height=384),
                examples=["resource/coloring/sample (1).png", "resource/coloring/sample (2).png"])
    app2 = gr.Interface(fn=img2img, 
                inputs=gr.Image(type="pil"),
                outputs=gr.Image(type="pil").style(width=256, height=384))
    app3 = gr.Interface(fn=text2img, inputs=gr.Image(type="pil"), outputs="text", examples=["resource/diary/sample.jpg"])
    demo = gr.TabbedInterface(title=md, interface_list=[app1, app2, app3], tab_names=["coloring book", "free drawing","diary"])
    demo.launch()


if __name__ == "__main__":
    run_app()
    #init_image = Image.open("resource/sample (1).png").convert("RGB")
    #img2img(init_image)
    

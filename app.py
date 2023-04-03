import gradio as gr
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

def img2img(img):
    device = "cuda"
    #model_id_or_path = "runwayml/stable-diffusion-v1-5"
    model_id_or_path = "model/dump"
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
    gr.Interface(fn=img2img, 
                inputs=gr.Image(type="pil"),
                outputs=gr.Image(type="pil").style(width=256, height=384),
                examples=["resource/coloring/sample (1).png", "resource/coloring/sample (2).png"]).launch(debug=True)
    demo.launch(share=True)

if __name__ == "__main__":
    run_app()
    #init_image = Image.open("resource/sample (1).png").convert("RGB")
    #img2img(init_image)
    
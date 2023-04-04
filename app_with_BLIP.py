import sys
sys.path.append("./BLIP")

import gradio as gr
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from BLIP.BLIP_infer import inference_BLIP

from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
    UnCLIPScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler,
)


def img2img(img):

    init_image = img
    init_image = init_image.resize((512, 768))

    caption = inference_BLIP(init_image)

    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config, use_karras_sigmas=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        scheduler = scheduler
    )
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(device)

    print(pipe.scheduler)

    generator = torch.manual_seed(3306892559)

    prompt = "masterpiece, best quality, ultra-detailed," + caption + ", illustration"
    negative_prompt = "worst quality, low quality, realistic, text, title, logo, signature, abs, muscular, rib, depth of field, bokeh, blurry, greyscale, monochrome"
    print("eval")
    print("prompt: " + prompt)
    print("negative_prompt: " + negative_prompt)
    images = pipe(prompt=prompt,negative_prompt=negative_prompt, image=init_image, generator=generator, num_inference_steps=40, guidance_scale=7).images

    return images[0]



def run_app():
    scheduler_options = ['Placeholder A', 'Placeholder B', 'Placeholder C']

    def closest_match(x):
        return x + ": The Definitive Edition"

    def Dropdown_list(x):
        new_options = [*scheduler_options, x + " Remastered", x + ": The Remake", x + ": Game of the Year Edition",
                       x + " Steelbook Edition"]
        return gr.Dropdown.update(choices=new_options)

    def Recommend_new(x):
        return x + ": Highest Cosine Similarity"


    # demo = gr.Interface(fn=img2img,
    #          inputs=gr.Image(type="pil"),
    #          outputs=gr.Image(type="pil").style(width=256, height=384),
    #          examples=["resource/free_sketch/color_sketch.JPG", "resource/coloring/sample (1).png",
    #                    "resource/coloring/sample (2).png"]).launch(debug=True)

    demo = gr.Blocks()
    with demo:

        with gr.Row():
            image_file = gr.Image(type="filepath")

        with gr.Row():


        text_input = gr.Textbox(label="Search bar")
        b1 = gr.Button("Generate")

        # text_options = gr.Dropdown(scheduler_options, label="Top 5 options")
        # b2 = gr.Button("Provide Additional options")
        #
        # new_title = gr.Textbox(label="Here you go!")
        # b3 = gr.Button("Recommend a new title")

        b1.click(closest_match, inputs=text_input, outputs=text_options)
        # b2.click(Dropdown_list, inputs=text_input, outputs=text_options)
        # b3.click(Recommend_new, inputs=text_options, outputs=new_title)
        # text_options.update(interactive=True)


    demo.launch(share=True)

if __name__ == "__main__":
    run_app()

    


import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline
from BLIP.BLIP_infer import BlipInfer
from .google_translator import GoogleTranslator
from .open_ai import OpenAIAPI

# OCR
import numpy as np
import cv2
import re
import easyocr
import json
import gradio as gr

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

class ImageGen:
    def __init__(self) -> None:
        self.device = "cuda"
        #model_id_or_path = "runwayml/stable-diffusion-v1-5"
        #model_id_or_path = "../model/dosmix/"
        model_id_or_path = "../model/dump2/"
        #model_id_or_path = "model/dump/"
        self.load_model(self.device, model_id_or_path)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.blip_infer = BlipInfer()
        self.translator = GoogleTranslator()
        self.openai_api = OpenAIAPI()
        self.reader = easyocr.Reader(['ko','en'], gpu=True) # need to run only once to load model into memory 
       

    def load_model(self, device, model_id_or_path):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
        #scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
        #scheduler = PNDMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")

        # vae
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
        
        
        # img2img
        self.i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16,  scheduler = scheduler)
        self.i2i_pipe.safety_checker = lambda images, clip_input: (images, False)
        self.i2i_pipe = self.i2i_pipe.to(device)
        self.i2i_pipe.vae = vae


        
        # text2img
        self.t2i_pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, scheduler = scheduler)
        #self.t2i_pipe = StableDiffusionKDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        #self.t2i_pipe.set_scheduler('sample_dpmpp_2m')
        self.t2i_pipe.safety_checker = lambda images, clip_input: (images, False)
        self.t2i_pipe = self.t2i_pipe.to(device)
        self.t2i_pipe.vae = vae

        # upscaler
        model_id = "stabilityai/sd-x2-latent-upscaler"
        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.upscaler.to(device)

    def img2img(self, img, filename = None):

        def read_img2img_json(path):
            # './resource/coloring/coloring.json'
            with open(path, 'r') as file:
                img2img_json_data = json.load(file)
            return img2img_json_data

        if filename != None:
            img2img_json_path = './resource/coloring/coloring.json'
            img2img_json_data = read_img2img_json(img2img_json_path)
            prompt, negative_prompt, seed = img2img_json_data[filename]['prompt'], \
                                            img2img_json_data[filename]['negative prompt'], \
                                            img2img_json_data[filename]['seed']
            img_size = img2img_json_data[filename]['size']
            strength = img2img_json_data[filename]['strength']
            num_inference_steps = img2img_json_data[filename]['num_inference_steps']
            guidance_scale = img2img_json_data[filename]['guidance_scale']
        else:
            prompt = "masterpiece, best quality, ultra-detailed, illustration"
            negative_prompt = "nsfw, worst quality, low quality, jpeg artifacts, depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare, greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature"
            seed = 0
            img_size = [512,512]
            strength = 0.5
            num_inference_steps = 30
            guidance_scale = 7

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        init_image = img
        init_image = init_image.resize(img_size)

        images = self.i2i_pipe(prompt=prompt, strength = strength, negative_prompt=negative_prompt,
                               image=init_image, generator=self.generator,
                               num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

        # low_res_latents = self.i2i_pipe(prompt=prompt, strength = 0.5, negative_prompt=negative_prompt,
        #                          image=init_image, generator=self.generator,
        #                          num_inference_steps=40, guidance_scale=7, output_type="latent").images
        # upscaled_image = self.upscaler(
        #     prompt=prompt,
        #     image=low_res_latents,
        #     num_inference_steps=40,
        #     guidance_scale=7,
        #     generator=self.generator,
        #     output_type="pil"
        #     ).images[0]
        #
        # return upscaled_image
        return images[0]
    
    def img2img_clip(self, img):
        init_image = img
        init_image = init_image.resize((784, 784))
        caption = self.blip_infer.inference(init_image)
        prompt = "masterpiece, best quality, ultra-detailed," + caption + ", illustration"
        negative_prompt = "nsfw, worst quality, low quality, jpeg artifacts, depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare, greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature"
        #generator = torch.manual_seed(-1)
        images = self.i2i_pipe(prompt=prompt,negative_prompt=negative_prompt, image=init_image, generator=self.generator, num_inference_steps=40, guidance_scale=7).images

        return caption, images[0]
    
    def ocr(self, split_imgs):
        # TODO: ocr from split_imgs to result
        # str_list = []
        str_list =  str()
        # str_list =  []
        for image in split_imgs:
            numpy_image=np.array(image)
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            result, score = self.reader.readtext(opencv_image)[0][1], self.reader.readtext(opencv_image)[0][2]
            if score < 0.4:
                # str_list.append(' ')
                str_list += ' '
            else:
                # str_list.append(result)
                str_list += result
        str_list= re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s]', "", str_list)
        # result = '오늘은, 재밌었다.'
        result = str_list.lstrip().rstrip()
        return result

    def text2img(self, img):
        input_rows = 5
        input_cols = 10
        start_point = (82,702)
        box = 63
        margin = 0
        #generator = torch.manual_seed(-1)
        split_imgs = []
        for rows in range(input_rows):
            for cols in range(input_cols):
                left = cols*box+start_point[0]+margin
                right = left+box
                top = rows*box+start_point[1]+margin
                bottom = top+box
                split_img = img.crop((left, top, right, bottom))
                #split_img.save("output/{}_{}.jpg".format(rows, cols))
                split_imgs.append(split_img)
        
        #im1 = im.crop((left, top, right, bottom))
        result_ocr = self.ocr(split_imgs)
        #print(result_ocr)
        result_translation = self.translator.translate(result_ocr)
        #result_prompts_list = self.openai_api.run(result_translation)
        #sample = "I wore my rain boots to school today, and I thought I'd wear them again after school, but it stopped raining, so it wasn't so good. I hope it rains tomorrow."
        #result_prompt_list = self.openai_api.run(sample)
        answer = '- A digital painting of a school hallway with a student wearing their rain boots, hoping for rain, muted color palette, low lighting, artstation, highly detailed, sharp focus.\n- A moody illustration of a student looking out a window, rain droplets sliding down the glass, with their rain boots by the door, artstation, trending, low angle, sharp focus, high detail, dark color scheme.\n- A whimsical digital portrait of a student sitting in a park with their rain boots on, surrounded by flowers and a rainbow overhead, art by loish and rossdraws, intricate details, pastel color scheme, sharp focus, artstation.\n- An emotional piece of a student standing outside in the sun with their rain boots on, staring up at the clear sky, their disappointment palpable, art by Sam Yang and trending on deviantart, intricate details, sharp lines, glossy finish, digital art.\n- An atmospheric digital painting of a student walking home from school with their rain boots, the sun setting behind them, casting long shadows and a warm glow over everything, artstation, concept art, smooth, sharp focus, warm color palette.'
        answer = answer.replace('- ', '')
        result_prompt_list = answer.split('\n')
        result_image_list = []
        for result_prompt in result_prompt_list:
            prompt = "masterpiece, best quality, ultra-detailed, upperbody, " + result_prompt + ", illustration"
            negative_prompt = "nsfw, worst quality, low quality, jpeg artifacts, depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare, greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature"
            images = self.t2i_pipe(prompt=prompt,negative_prompt=negative_prompt, generator=self.generator, num_inference_steps=40, guidance_scale=7).images
            result_image_list.append(images[0])
            #low_res_latents = self.t2i_pipe(prompt=prompt,negative_prompt=negative_prompt, generator=self.generator, num_inference_steps=40, guidance_scale=7, output_type="latent").images
            '''
            low_res_latents = self.t2i_pipe(prompt=prompt,negative_prompt=negative_prompt, generator=self.generator, num_inference_steps=40, guidance_scale=7, use_karras_sigmas=True).images
            upscaled_image = low_res_latents[0]
            upscaled_image = self.upscaler(
                prompt=prompt,
                image=low_res_latents,
                num_inference_steps=30,
                guidance_scale=7,
                generator=self.generator,
                output_type="pil"
            ).images[0]
            result_image_list.append(upscaled_image)
            '''
            
            break
        return result_ocr+'\n'+result_translation, result_prompt_list[0], result_image_list[0]
    
if __name__ == "__main__":
    img_gen = ImageGen()
    from PIL import Image
    im = Image.open("resource/diary/sample.jpg") 
    result_prompt_list, result_image_list = img_gen.text2img(im)
    for i, img in enumerate(result_image_list):
         img.save("output/{}.jpg".format(i))
    

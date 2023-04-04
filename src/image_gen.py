
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from BLIP.BLIP_infer import BlipInfer
from .google_translator import GoogleTranslator
from .open_ai import OpenAIAPI

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
        device = "cuda"
        model_id_or_path = "runwayml/stable-diffusion-v1-5"
        #model_id_or_path = "model/dump"
        # torch_dtype=torch.float16
        # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        self.load_model(device, model_id_or_path)
        self.blip_infer = BlipInfer()
        self.translator = GoogleTranslator()
        self.openai_api = OpenAIAPI()
       

    def load_model(self, device, model_id_or_path):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, scheduler = scheduler)
        self.pipe.safety_checker = lambda images, clip_input: (images, False)
        #self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config, use_karras_sigmas=True)
        self.pipe = self.pipe.to(device)

    def img2img(self, img):
        init_image = img
        init_image = init_image.resize((512, 768))
        generator = torch.manual_seed(3306892559)
        prompt = "masterpiece, best quality, ultra-detailed, illustration,  1princess"
        negative_prompt = "worst quality, low quality, realistic, text, title, logo, signature, abs, muscular, rib, depth of field, bokeh, blurry, greyscale, monochrome"
        images = self.pipe(prompt=prompt,negative_prompt=negative_prompt, image=init_image, generator=generator, num_inference_steps=40, guidance_scale=7).images
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
    
    
    def img2img_clip(self, img):

        init_image = img
        init_image = init_image.resize((512, 512))
        caption = self.blip_infer.inference(init_image)
        generator = torch.manual_seed(3306892559)

        prompt = "masterpiece, best quality, ultra-detailed," + caption + ", illustration"
        negative_prompt = "worst quality, low quality, realistic, text, title, logo, signature, abs, muscular, rib, depth of field, bokeh, blurry, greyscale, monochrome"
        print("eval")
        print("prompt: " + prompt)
        print("negative_prompt: " + negative_prompt)
        images = self.pipe(prompt=prompt,negative_prompt=negative_prompt, image=init_image, generator=generator, num_inference_steps=40, guidance_scale=7).images

        return images[0]
    
    
    def ocr(self, split_imgs):
        # TODO: ocr from split_imgs to result
        result = '오늘은, 재밌었다.'
        return result

    def text2img(self, img):
        input_rows = 5
        input_cols = 10
        start_point = (82,702)
        box = 63
        margin = 0
        generator = torch.manual_seed(3306892559)
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
        result_ocr = self.ocr(split_imgs)
        result_translation = self.translator.translate(result_ocr)
        #result_prompts_list = self.openai_api.run(result_translation)
        #sample = "I wore my rain boots to school today, and I thought I'd wear them again after school, but it stopped raining, so it wasn't so good. I hope it rains tomorrow."
        #result_prompt_list = self.openai_api.run(sample)
        answer = '- A digital painting of a school hallway with a student wearing their rain boots, hoping for rain, muted color palette, low lighting, artstation, highly detailed, sharp focus.\n- A moody illustration of a student looking out a window, rain droplets sliding down the glass, with their rain boots by the door, artstation, trending, low angle, sharp focus, high detail, dark color scheme.\n- A whimsical digital portrait of a student sitting in a park with their rain boots on, surrounded by flowers and a rainbow overhead, art by loish and rossdraws, intricate details, pastel color scheme, sharp focus, artstation.\n- An emotional piece of a student standing outside in the sun with their rain boots on, staring up at the clear sky, their disappointment palpable, art by Sam Yang and trending on deviantart, intricate details, sharp lines, glossy finish, digital art.\n- An atmospheric digital painting of a student walking home from school with their rain boots, the sun setting behind them, casting long shadows and a warm glow over everything, artstation, concept art, smooth, sharp focus, warm color palette.'
        answer = answer.replace('- ', '')
        result_prompt_list = answer.split('\n')
        result_image_list = []
        for result_prompt in result_prompt_list:
            prompt = "masterpiece, best quality, ultra-detailed," + result_prompt + ", illustration"
            negative_prompt = "worst quality, low quality, realistic, text, title, logo, signature, abs, muscular, rib, depth of field, bokeh, blurry, greyscale, monochrome"
            images = self.pipe(prompt=prompt,negative_prompt=negative_prompt, generator=generator, num_inference_steps=40, guidance_scale=7).images
            result_image_list.append(images[0])
        return result_prompt_list, result_image_list
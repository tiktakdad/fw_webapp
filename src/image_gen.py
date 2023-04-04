
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
#from BLIP.BLIP_infer import inference_BLIP

# OCR
import numpy as np
import cv2
import re
import easyocr
reader = easyocr.Reader(['ko','en'], gpu=True) # need to run only once to load model into memory 


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
       

    def load_model(self, device, model_id_or_path):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
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
    
    
    def ocr(self, split_imgs):
        # TODO: ocr from split_imgs to result
        # str_list = []
        str_list =  str()
        # str_list =  []
        for image in split_imgs:
            numpy_image=np.array(image)
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            result, score = reader.readtext(opencv_image)[0][1], reader.readtext(opencv_image)[0][2]
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
        return self.ocr(split_imgs)
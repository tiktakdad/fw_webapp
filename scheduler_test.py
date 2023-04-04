from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
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
import torch

# BLIP 들어가야 되는 곳


# structure model 정의
model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "/home/ktva/stable-diffusion-webui/models/Stable-diffusion/chilloutmix-Ni.safetensors"
#model_id = "CompVis/stable-diffusion-v1-4"

# 스케줄러 정의
ddpm = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# 모델 load
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline.to("cuda")

prompt = "masterpieces, best quality, tree, storybook illustration"
negative_prompt = ""
generator = torch.manual_seed(10000) # Seed 지정
low_res_latents = pipeline(prompt, negative_prompt = negative_prompt, generator=generator, output_type="latent").images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]
print(type(image))
image.save("./images/a1.png")



'''
model_id = "stabilityai/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
upscaler.to("cuda")


upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]

upscaled_image.save("./images/a2.png")
'''
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder


class BlipInfer:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 384
        self.load_model()

    def load_model(self):
        model_url = './BLIP/pth/model_base.pth'
        self.model = blip_decoder(pretrained=model_url, image_size=self.image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(self.device)

    def inference(self, raw_image):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
        image = transform(raw_image).unsqueeze(0).to(self.device)   

        with torch.no_grad():
            # beam search
            caption = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        return caption[0]


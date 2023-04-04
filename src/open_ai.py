import openai
import json

openai.api_key = "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"

class OpenAIAPI:
    def __init__(self) -> None:
       pass
    
    def run(self, text):
        print(text)
        prompt_initial = f"Stable Diffusion is an AI art generation model similar to DALLE-2.\
                Here are some prompts for generating art with Stable Diffusion.\
                Example:\
                - portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration\
                - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration\
                - ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image\
                - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation\
                - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm\
                - athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration\
                - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration\
                - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration\
                - portrait of beautiful happy young ana de armas, ethereal, realistic anime, trending on pixiv, detailed, clean lines, sharp lines, crisp lines, award winning illustration, masterpiece, 4k, eugene de blaas and ross tran, vibrant color scheme, intricately detailed\
                - A highly detailed and hyper realistic portrait of a gorgeous young ana de armas, lisa frank, trending on artstation, butterflies, floral, sharp focus, studio photo, intricate details, highly detailed, alberto seveso and geo2099 style\
                Prompts should be written in English, excluding the artist name, and include the following rule:\
                - Follow the structure of the example prompts. This means Write a description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more, excluding the artist name, separated by commas. place a extra commas.\
                I want you to write me a list of detailed prompts exactly about the IDEA follow the rule at least 5 every time.\
                IDEA: %s" % (text)
        self.completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"%s" % prompt_initial},
            ]
        )
        answer = self.completion['choices'][0]['message']['content'].strip()
        answer = answer.replace('- ', '')
        prompts = answer.split('\n')
        return prompts

    



if __name__ == "__main__":
    
    api = OpenAIAPI()
    answer = api.run("I wore my rain boots to school today, and I thought I'd wear them again after school, but it stopped raining, so it wasn't so good. I hope it rains tomorrow.")
    print('answer:',answer)
    
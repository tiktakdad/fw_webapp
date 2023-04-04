import easyocr
import os


reader = easyocr.Reader(['ko','en'], gpu=True) # need to run only once to load model into memory 

path = 'test_image'
image_list = os.listdir(path)
image_list.sort()

for image in image_list:
    print('image name: ', image)
    image_path = os.path.join(path, image)

    result = reader.readtext(image_path)
    print(result)

import os
from PIL import Image
import numpy as np
import glob

def resize_image(src_path: str, dst_path: str):
    cwd = os.getcwd()
    image_path = src_path + str('/*')
    all_images = glob.glob(os.path.join(image_path))
    print(all_images)
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    for image_path in all_images:
        image = Image.open(image_path)
        new_image = image.resize((256, 256))
        new_image_path = os.path.join(dst_path, image_path.split('/')[-1][:-4] + str('.png'))
        print('new_image_path: ', new_image_path)
        new_image.save(new_image_path)


if __name__ == '__main__':
    resize_image(src_path='/home/sky/mcsp/CORE1/src/data/lsun/test/real', dst_path='/home/sky/mcsp/CORE1/src/data/lsun/test/real1')
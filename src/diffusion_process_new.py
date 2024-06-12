from typing import Union, Tuple, Optional
import os
# import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageChops, ImageMath
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
import numpy as np
import glob
from collections import Counter

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None, img_size = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if img_size is not None:
        pil_img = pil_img.resize((img_size, img_size))
    print("img_sizeeeeeeeeeeeeeeeeee", pil_img.size)
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

def diff_img(img1, img2):
    w, h = img1.size
    diff = ImageChops.difference(img1, img2)
    diff_np = np.array(diff)
    err = np.sum(diff_np**2)
    mse = err/(float(h*w))
    return diff, mse

@torch.no_grad()
def ddim_inversion(root_path: str, num_steps: int = 50, verify: Optional[bool] = False, img_size = None) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    cwd = os.getcwd()
    save_path = root_path + str("-dd")
    real_path = os.path.join(save_path, "original")
    recon_path = os.path.join(save_path, "reconstruct")
    diff_path = os.path.join(save_path, "diff")
    print("root_pathhh", root_path, save_path)
    print("real_pathhhh", real_path)    

    if not os.path.isdir(real_path):
        os.makedirs(real_path)
    if not os.path.isdir(recon_path):
        os.makedirs(recon_path)
    if not os.path.isdir(diff_path):
        os.makedirs(diff_path)

    exist_imgs = os.listdir(real_path)
    exist_imgs = [str(ei[:-8]) for ei in exist_imgs]
    imgs_b = glob.glob(root_path+str("/*"))
    imgs = []
    # exist_count = Counter(exist_imgs)
    for img in imgs_b:
        fn = img.split("\\")[-1][:-4]
        if str(fn) not in exist_imgs:
            print(fn)
            imgs.append(img)

    print('imagessssssss', imgs, len(imgs), len(exist_imgs))
    print('exist imagessssssss', exist_imgs)

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')

    for imgname in imgs:
        pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                                                    scheduler=inverse_scheduler,
                                                    safety_checker=None,
                                                    torch_dtype=dtype)
        pipe.to(device)
        vae = pipe.vae
        input_img = load_image(imgname, img_size).to(device=device, dtype=dtype)
        latents = img_to_latents(input_img, vae)

        inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                            width=input_img.shape[-1], height=input_img.shape[-2],
                            output_type='latent', return_dict=False,
                            num_inference_steps=num_steps, latents=latents)

        # verify
        if verify:
            pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
            latent_img = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                        num_inference_steps=num_steps, latents=latents)
            image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                        num_inference_steps=num_steps, latents=inv_latents)
            # fig, ax = plt.subplots(1, 2)
            # original image
            in_img = tvt.ToPILImage()(input_img[0])
            # reconstructed image
            out_img = image.images[0]
            # difference between original image and reconstructed image
            diff, mse = diff_img(in_img, out_img)
            # orig_path_out = os.path.join(real_path, imgname.split("/")[-1][:-4]+ str("test-ori.png"))
            # recon_path_out = os.path.join(recon_path, imgname.split("/")[-1][:-4] + str("test-rec.png"))
            # diff_path_out = os.path.join(diff_path, imgname.split("/")[-1][:-4] + str("test-diff.png"))

            orig_path_out = os.path.join(real_path, imgname.split("/")[-1][:-4]+ str("sp1-ori.png"))
            recon_path_out = os.path.join(recon_path, imgname.split("/")[-1][:-4] + str("sp1-rec.png"))
            diff_path_out = os.path.join(diff_path, imgname.split("/")[-1][:-4] + str("sp1-diff.png"))

            in_img.save(orig_path_out)
            out_img.save(recon_path_out)
            diff.save(diff_path_out)
            print('mse of 2 images: ', mse)
            # ax[0].imshow(tvt.ToPILImage()(input_img[0]))
            # ax[1].imshow(image.images[0])
            # ax[1].imshow(latent_img)
            # plt.show()
    return inv_latents

if __name__ == '__main__':
    ddim_inversion(root_path='/home/sky/mcsp/ddim/data/lsun/test/ldm', num_steps=250, verify=True, img_size=256)
    # ddim_inversion(root_path='/home/sky/mcsp/CORE1/src/data/celebahq/test/test/midjourney', num_steps=250, verify=True, img_size=None)
    # ddim_inversion(root_path='/home/sky/mcsp/CORE1/src/data/Celeb-DF-v2-imgs/train/real', num_steps=250, verify=True, img_size=None)
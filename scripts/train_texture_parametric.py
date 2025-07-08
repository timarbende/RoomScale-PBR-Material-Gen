import os
import argparse

import torch
import torchvision

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(".")
from models.pipeline.texture_pipeline import TexturePipeline

from models.pipeline.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler
from models.utils.load_image import load_ldr_image

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/optimize_texture.yaml")
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_step", type=int, default=0)
    parser.add_argument("--texture_size", type=int, default=4096)

    # only with template
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--scene_id", type=str, default="", help="<house_id>/<room_id>")

    args = parser.parse_args()

    if args.stamp is None:
        setattr(args, "stamp", "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "debug"))

    return args

def init_config(args):
    config = OmegaConf.load(args.config)

    # template
    if len(args.log_dir) != 0 and len(args.prompt) != 0 and len(args.scene_id) != 0:
        print("=> filling template with following arguments:")
        print("   log_dir:", args.log_dir)
        print("   prompt:", args.prompt)
        print("   scene_id:", args.scene_id)

        config.log_dir = args.log_dir
        config.prompt = args.prompt
        config.scene_id = args.scene_id

    return config


def init_scenetex_pipeline(
        config,
        stamp,
        device=DEVICE,
        inference_mode=False
    ):
    pipeline = TexturePipeline(
        config=config,
        stamp=stamp,
        device=device
    ).to(device)

    pipeline.configure(inference_mode=inference_mode)

    return pipeline

def init_rgb2x_pipeline(config):
    pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=config.cache_dir,
    ).to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    return pipeline

def parameterize_texture(rgb2x_pipeline):
    num_inference_steps = 50
    generator = torch.Generator(device="cuda").manual_seed(0)
    photo_path = "./texture_29900.png"
    photo = load_ldr_image(photo_path, from_srgb=True).to("cuda")

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    return_list = []
    for aov_name in required_aovs:
        prompt = prompts[aov_name]
        generated_image = rgb2x_pipeline(
            prompt=prompt,
            photo=photo,
            num_inference_steps=num_inference_steps,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=[aov_name],
        ).images[0][0]

        # images is Union[List[PIL.Image.Image], np.ndarray] where List[PIL.Image.Image] is the list of images and np.ndarray is list of whether corresponding image contains nsfw
        # we take the first image in the list in the union

        generated_image = torchvision.transforms.Resize((old_height, old_width))(generated_image)

        return_list.append((generated_image, aov_name))

    return return_list

def save_images(images):
    for image, name in images:
        image.save(f"{name}.png")

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    args = init_args()

    if len(args.checkpoint_dir) < 1:
        print("checkpoint missing")
        sys.exit()

    config = init_config(args)

    scenetex_pipeline = init_scenetex_pipeline(config=config, stamp=args.stamp, inference_mode=True)
    rgb2x_pipeline = init_rgb2x_pipeline(config=config)

    print("inference mode")
    print("prompt:", config.prompt)
    
    scenetex_pipeline.load_checkpoint(args.checkpoint_dir, args.checkpoint_step)
    
    scenetex_pipeline.inference(args.checkpoint_dir, args.checkpoint_step, args.texture_size)
    print("baked texture generated")

    texture_aovs=parameterize_texture(rgb2x_pipeline=rgb2x_pipeline)

    save_images(texture_aovs)
import os
import argparse

import torch
import torchvision

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

import lovely_tensors as lt
lt.monkey_patch()

import sys
sys.path.append(".")
from models.pipeline.parametric_texture_pipeline import TexturePipeline
from torchvision.utils import save_image

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/optimize_texture.yaml")
    args = parser.parse_args()

    print("=> loading config file {}...".format(args.config))

    config = OmegaConf.load(args.config)
    return config


def init_pipeline(
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

    pipeline.configure()

    return pipeline

from models.modules import TextureMesh, Studio
import numpy as np

def debug_resize(photo):
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

    resized_photo = torchvision.transforms.Resize((new_height, new_width))(photo)
    torchvision.transforms.ToPILImage()(resized_photo).save("resized.png")

def debug_render_all_views():
    cameras_count = 10
    studio = Studio(config, DEVICE)
    texture_mesh = TextureMesh(config, DEVICE)

    for camera_id in range(cameras_count):
        Rs, Ts, fovs, _, image_path = studio.sample_cameras(camera_id, config.batch_size, random_cameras=False)            
        camera = studio.set_cameras(Rs, Ts, fovs)

        renderer = studio.set_renderer(camera, config.render_size)
        _, fragments = studio.render(renderer, texture_mesh.mesh, texture_mesh.texture, None, None, None, True)
        _, relative_depth = studio.get_relative_depth_map(fragments.zbuf)
        relative_depth = relative_depth / 255.0

        image_path = image_path.split("/")[-1]

        save_image(relative_depth, "{}.png".format(image_path))

def debug_render():
    texture_mesh = TextureMesh(config, DEVICE)

    decoded_texture_not_normalized = texture_mesh.texture[0].permute(2, 0, 1)
    torchvision.transforms.ToPILImage()(decoded_texture_not_normalized).save("original_texture.png")
    decoded_texture = (decoded_texture_not_normalized / 2 + 0.5).clamp(0, 1)
    decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture).convert("RGB")
    decoded_texture.save("debug_texture.png")


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["WANDB_START_METHOD"] = "thread"

    torch.backends.cudnn.benchmark = True

    config = init_config()

    #TODO (all):
    # - log frequency weighting (parametric pipeline fit)

    #TODO: ideas if time
    # - oversmoothing fix: run one aov and one view diffusion with random noise, then run again with fixed noise. Compare results
    #       - for multiple views: we have a noise texture, we render it from the views
    #       - to get the latent dimension size (encoded size) instead of encoding we downsize the noise (because encoder cannot encode noise)

    # results needed for thesis
    # - teaser, fancy visualisation: albedo, roughness, metallic rendered in blender, add relighting
    # - comparision on synthetic data (kitchen_hq) with other methods (qualitative and quantitative results like the one Peter sent) (all aov for at least 3-4 room (kitchen, bedroom, living room))
    # - comparision on real data(scannet) with other methods (qualitative results)
    # - ablations: intermediate results while developing the method (1-2 steps)
    #       - latent space -> image space loss
    #       - with / without guidance scale
    #       - texture initialization
    # generated data method will be application ("this method can also be used to first generate texture with baked in lights and then decomposition that texture into its parameters")

    # TODO: inference_mode = hasattr(config, "checkpoint_dir") and len(config.checkpoint_dir) > 0

    inference_mode = False
    
    prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        } 

    stamp = "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "debug")

    for aov in prompts.keys():
        config.aov = aov
        config.prompt = prompts[aov]
        pipeline = init_pipeline(
            config=config, 
            stamp=stamp,
            inference_mode=inference_mode
        )
        print("=> start training", aov, "...")
        with torch.autograd.set_detect_anomaly(True):
            pipeline.fit()
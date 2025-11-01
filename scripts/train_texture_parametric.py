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
from models.pipeline.parametric_texture_pipeline_frequency import TexturePipeline
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

    config = OmegaConf.load(args.config)
    return config


def init_pipeline(
        config,
        stamp,
        aov,
        prompt,
        device=DEVICE,
        inference_mode=False
    ):
    pipeline = TexturePipeline(
        config=config,
        stamp=stamp,
        device=device,
        aov=aov,
        prompt = prompt
    ).to(device)

    pipeline.configure()

    return pipeline

from models.modules import TextureMesh, Studio
import numpy as np


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["WANDB_START_METHOD"] = "thread"

    torch.backends.cudnn.benchmark = True

    print("=> loading config file...")
    config = init_config()

    def render_all_views():
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

    render_all_views()

    #TODO (all):
    # - metallic roughness channel fix (guidance compute_image_space_sds_loss)
    # - fix scannet: resize inputs (parametric_texture_pipeline fit, forward; guidance compute_image_space_sds_loss)
    #       - change conditioning image resolution (768*x) (rgbx preprocess_image) must be dividible by 8 (682 wont work)
    # - ray-based-update: remove random pixels from loss by random mask (guidance compute_image_space_sds_loss)

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
    
    '''
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
        pipeline = init_pipeline(
            config=config, 
            stamp=stamp,
            aov=aov,
            prompt=prompts[aov],
            inference_mode=inference_mode
        )
        print("=> start training", aov, "...")
        with torch.autograd.set_detect_anomaly(True):
            pipeline.fit()
    '''
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
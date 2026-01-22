import argparse
import torch
from omegaconf import OmegaConf
from datetime import datetime

import lovely_tensors as lt
lt.monkey_patch()

import sys
sys.path.append(".")
from models.pipeline.view_count_texture_pipeline import TexturePipeline

import os

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
    ):
    pipeline = TexturePipeline(
        config=config,
        stamp=stamp,
        device=device
    ).to(device)

    pipeline.configure()

    return pipeline

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    config = init_config()
    config.use_wandb = False
    config.aov=""

    inference_mode = False

    stamp = "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "debug")

    pipeline = init_pipeline(
            config=config, 
            stamp=stamp
        )
    print("=> start training view-count texture...")
    with torch.autograd.set_detect_anomaly(True):
        pipeline.fit()
        
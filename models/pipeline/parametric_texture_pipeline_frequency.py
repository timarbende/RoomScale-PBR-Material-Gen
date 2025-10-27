import random
import wandb
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

import pytorch_lightning as pl

# mat
import matplotlib.pyplot as plt

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from omegaconf import OmegaConf

from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from copy import deepcopy
from pathlib import Path

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import interpolate_face_attributes

from models.pipeline.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler
from models.utils.load_image import load_ldr_image

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)

from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes

from mpl_toolkits.mplot3d import Axes3D

# customized
import sys
sys.path.append("./lib")
from models.modules import TextureMesh, Studio
from models.modules.guidance_parametric import Guidance
import cv2

from PIL import Image

'''
texture pipeline for creating the frequency texture (which point of geometry has been seen how many times)
'''
class TexturePipeline(nn.Module):
    def __init__(self, 
        config,
        stamp,
        device
    ): 
        
        super().__init__()

        self.config = config
        self.stamp = stamp

        self.prompt = config.prompt + ", " + config.a_prompt if config.a_prompt else config.prompt
        self.n_prompt = config.n_prompt
        
        self.device = device
        self.weights_dtype = torch.float16 if self.config.enable_half_precision else torch.float32
        print("=> Use precision: {}".format(self.weights_dtype))

        pl.seed_everything(self.config.seed)


    """call this after to(device)"""
    def configure(self):
        # 3D assets
        self._init_mesh()

        # studio
        self._init_studio()

        self._init_guidance()

        # optimization
        self._configure_optimizers()

    def _init_studio(self):
        self.studio = Studio(self.config, self.device)

    def _init_mesh(self):
        self.texture_mesh = TextureMesh(self.config, self.device)


    def _init_guidance(self):
        self.guidance = Guidance(self.config, self.device)

    def _get_texture_parameters(self):
        if "hashgrid" not in self.config.texture_type:
            texture_params = [self.texture_mesh.texture]

            if self.config.use_background: 
                texture_params += [self.texture_mesh.background_texture]

        else:
            texture_params = [p for p in self.texture_mesh.texture.parameters() if p.requires_grad]

            if self.config.use_background: 
                texture_params += [p for p in self.texture_mesh.background_texture.parameters() if p.requires_grad]

        # render function
        texture_params += [p for p in self.studio.render_func.parameters() if p.requires_grad]
        
        # anchor function
        if self.config.enable_anchor_embedding: 
            texture_params += [p for p in self.studio.anchor_func.parameters() if p.requires_grad]

        return texture_params

    def _get_guidance_parameters(self):
        return [p for p in self.guidance.unet_phi_layers.parameters() if p.requires_grad]

    def _configure_optimizers(self):
        texture_params = self._get_texture_parameters()

        self.texture_optimizer = SGD(texture_params, lr=1.0)

    @torch.no_grad()
    def inference(self, checkpoint_dir, checkpoint_step, texture_size):
        u, v = torch.arange(texture_size).to(self.device), torch.arange(texture_size).to(self.device)
        u, v = torch.meshgrid(u, v, indexing='ij')
        inputs = torch.stack([u, v]).permute(1, 2, 0).unsqueeze(0) / (texture_size - 1)

        texture = torch.zeros(texture_size, texture_size, 3)
        
        if self.config.enable_anchor_embedding: 
            instance_map = self.texture_mesh.instance_map[None, None, :, :]
            instance_map = F.interpolate(instance_map, (texture_size, texture_size), mode="nearest")
            instance_map = instance_map.permute(0, 1, 3, 2)
            instance_map = torch.flip(instance_map, dims=[3])

        for i in tqdm(range(texture_size)):
            r_inputs = inputs[:, i].float()
            r_inputs = self.studio.query_texture(r_inputs.unsqueeze(1), self.texture_mesh.texture)
            
            if self.config.enable_anchor_embedding: 
                r_inputs = self.studio.query_anchor_features(
                    self.texture_mesh.instance_anchors, 
                    self.texture_mesh.texture, 
                    r_inputs, 
                    instance_map[:, :, i]
                )

            r = self.studio.render_func(r_inputs)[0, 0]
            texture[i] = r.detach().cpu()
        
        texture = (texture / 2 + 0.5).clamp(0, 1)
        texture = texture.cpu()
        texture = texture.permute(1, 0, 2)
        texture = torch.flip(texture, dims=[0])

        assert texture.min() >= 0 and texture.max() <= 1

        texture = torchvision.transforms.ToPILImage()(texture.permute(2, 0, 1)).convert("RGB")
        texture.save(os.path.join(checkpoint_dir, "texture_{}.png".format(checkpoint_step)))

    def _prepare_mesh(self, inference):
        mesh = self.texture_mesh.mesh
        background_mesh = self.texture_mesh.background_mesh if self.config.use_background and not inference else None

        if self.config.batch_size > 1 and not inference:
            mesh = self.texture_mesh.repeat_meshes_as_batch(self.texture_mesh.mesh, self.config.batch_size)
            if background_mesh is not None: 
                background_mesh = self.texture_mesh.repeat_meshes_as_batch(self.texture_mesh.background_mesh, self.config.batch_size)
        
        # textures
        texture = self.texture_mesh.texture
        background_texture = self.texture_mesh.background_texture if self.config.use_background and not inference else None

        return mesh, texture, background_mesh, background_texture

    def forward(self, camera, inference=False, downsample=True, is_direct=False):
        renderer = self.studio.set_renderer(camera, self.config.render_size)

        mesh, texture, background_mesh, background_texture = self._prepare_mesh(inference)

        anchors = self.texture_mesh.instance_anchors if self.config.enable_anchor_embedding else None
 
        latents = self.studio.render(renderer, mesh, texture, background_mesh, background_texture, anchors, is_direct)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    def fit(self):
        pbar = tqdm(range(self.studio.num_cameras))

        for step, chosen_t in enumerate(pbar):
            Rs, Ts, fovs, _, _ = self.studio.sample_cameras(step, 1, random_cameras=False, inference=False) 
            cameras = self.studio.set_cameras(Rs, Ts, fovs)

            latents = self.forward(cameras, is_direct=("hashgrid" not in self.config.texture_type))

            # compute loss
            self.texture_optimizer.zero_grad()

            loss = self.guidance.compute_frequency_loss(latents)

            loss.backward()

            self.texture_optimizer.step()
            
            max_memory_allocated = torch.cuda.max_memory_allocated()
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {chosen_t}, GPU: {max_memory_allocated / 1024**3:.2f} GB')

            torch.cuda.empty_cache()

        #TODO: save in exr
        frequency_texture = self.texture_mesh.texture.squeeze(0).permute(2, 0, 1)
        frequency_texture = frequency_texture / torch.max(frequency_texture)
        frequency_texture = torchvision.transforms.ToPILImage(mode="RGB")(frequency_texture)
        frequency_texture.save(self.config.frequency_texture_path)
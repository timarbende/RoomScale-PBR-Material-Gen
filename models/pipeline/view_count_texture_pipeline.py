import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.optim import SGD
from tqdm import tqdm
import os

# customized
import sys
sys.path.append("./lib")
from models.modules import TextureMesh, Studio
from models.modules.guidance_parametric import Guidance

'''
texture pipeline for creating the view-count texture (which point of geometry has been seen how many times)
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
 
        latents, _ = self.studio.render(renderer, mesh, texture, background_mesh, background_texture, anchors, is_direct)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    def fit(self):
        pbar = tqdm(range(self.studio.num_cameras))

        for step, chosen_t in enumerate(pbar):
            Rs, Ts, fovs, _, _ = self.studio.sample_cameras(step, 1, random_cameras=False, inference=False) 
            cameras = self.studio.set_cameras(self.config.camera_type, Rs, Ts, fovs)

            latents = self.forward(cameras, is_direct=("hashgrid" not in self.config.texture_type))

            # compute loss
            self.texture_optimizer.zero_grad()

            loss = self.guidance.compute_frequency_loss(latents)

            loss.backward()

            self.texture_optimizer.step()
            
            max_memory_allocated = torch.cuda.max_memory_allocated()
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {chosen_t}, GPU: {max_memory_allocated / 1024**3:.2f} GB')

            torch.cuda.empty_cache()

        paths = self.config.frequency_texture_path.split("/")[:-1]
        directory = os.path.join(*paths)
        if not os.path.exists(directory):
            os.makedirs(directory)
        frequency_texture = self.texture_mesh.texture.squeeze(0).permute(2, 0, 1)
        frequency_texture = frequency_texture / torch.max(frequency_texture)
        frequency_texture = torchvision.transforms.ToPILImage(mode="RGB")(frequency_texture)
        frequency_texture.save(self.config.frequency_texture_path)
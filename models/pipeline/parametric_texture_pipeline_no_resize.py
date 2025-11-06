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

from torch.optim import Adam, AdamW
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

from datetime import datetime

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
    def configure(self, inference_mode=False):

        print("Using {}-space loss".format(self.config.loss_space))

        # 3D assets
        self._init_mesh()

        # studio
        self._init_studio()

        # instances
        self._init_anchors()

        self._init_guidance()

        # optimization
        self._configure_optimizers()
        self._init_logger()

        if self.config.enable_clip_benchmark:
            import open_clip
            self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def _init_studio(self):
        self.studio = Studio(self.config, self.device)

    def _init_mesh(self):
        self.texture_mesh = TextureMesh(self.config, self.device)

        if("frequency_texture_path" in self.config and
            self.config.frequency_texture_path is not None
           and os.path.exists(self.config.frequency_texture_path)):
            
            frequency_texture_img = Image.open(self.config.frequency_texture_path)
            frequency_texture = torchvision.transforms.ToTensor()(frequency_texture_img).permute(1, 2, 0).cuda()

            mesh = self.texture_mesh.mesh
        
            self.frequency_mesh = mesh.clone()
            self.frequency_mesh.textures = TexturesUV(
                maps=frequency_texture[None, ...],  # B, H, W, C
                faces_uvs= mesh.textures.faces_uvs_padded(),
                verts_uvs= mesh.textures.verts_uvs_padded(),
                sampling_mode="bilinear"
            )

    def _init_guidance(self):
        self.guidance = Guidance(self.config, self.device)

    def _init_anchors(self):
        if self.config.enable_anchor_embedding:
            self.texture_mesh.build_instance_map(self.studio, self.config.log_dir)
            self.texture_mesh.sample_instance_anchors(self.config.log_dir)
            self.studio.init_anchor_func(self.texture_mesh.num_instances)

    def _init_logger(self):
        #os.makedirs(self.config.log_dir, exist_ok=True)

        model_type = "controlnet" if "controlnet" in self.config.diffusion_type else "SD"

        if(self.config.use_wandb):
            wandb.login()
            wandb.init(
                project="SceneTex",
                name="kitchen_hq_origin_init_{}_{}".format(self.config.aov, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                #dir=self.config.log_dir
            )
        else:
            print("Not using WandB (set use_wandb to True in template.yaml to enable it)")

        '''
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.config, f=f)
        '''

        if(self.config.use_wandb and self.config.wandb_log_hyperparameters):
            wandb.log({
                "config/guidance_scale": self.config.guidance_scale,
                "config/image_guidance_scale": self.config.image_guidance_scale,
                "config/learning_rate": self.config.latent_lr,
                "config/render_size": self.config.render_size,
                "config/latent_texture_size": self.config.latent_texture_size,
                "config/loss_type": self.config.loss_type,
                "config/diffusion_type": self.config.diffusion_type,
            })

        self.avg_loss_sds = []

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

        self.texture_optimizer = AdamW(texture_params, lr=self.config.latent_lr)

    def _downsample(self, inputs, in_size, out_size, mode="direct", type_="interpolate"):
        if mode == "iterative":
            assert type_ != "encoder"
            down_size_list = []
            down_size = in_size
            num_max_down = in_size // out_size
            for _ in range(num_max_down):
                down_size = down_size // 2
                if down_size <= out_size:
                    down_size = out_size

                down_size_list.append(down_size)

                if down_size == out_size: break

            outputs = inputs
            for down_size in down_size_list:
                if type_ == "interpolate":
                    outputs = F.interpolate(outputs, size=(down_size, down_size), mode="bilinear")
                elif type_ == "avg_pool":
                    outputs = F.adaptive_avg_pool2d(outputs, (down_size, down_size))
                else:
                    raise ValueError("invalid downsampling type.")

        elif mode == "direct":
                if type_ == "interpolate":
                    outputs = F.interpolate(inputs, size=(out_size, out_size), mode="bilinear")
                elif type_ == "avg_pool":
                    outputs = F.adaptive_avg_pool2d(inputs, (out_size, out_size))
                elif type_ == "encoder":
                    # outputs = self.latent_encoder(inputs)
                    raise NotImplementedError
                else:
                    raise ValueError("invalid downsampling type.")
        else:
            raise ValueError("invalid downsampling mode.")

        return outputs

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
 
        latents, _ = self.studio.render(renderer, mesh, texture, background_mesh, background_texture, anchors, is_direct)
        latents = latents.permute(0, 3, 1, 2)
        
        #TODO: downsize latents to also 768 * 512

        not_encoded_latents = latents
        not_encoded_latents = (not_encoded_latents / 2 + 0.5).clamp(0, 1)

        if downsample:
            if self.config.downsample == "vae":
                latents = self.guidance.encode_latent_texture(latents, self.config.deterministic_vae_encoding)
            elif self.config.downsample == "interpolate":
                latents = F.interpolate(latents, (self.config.latent_size, self.config.latent_size), mode="bilinear", align_corners=False)
            else:
                raise ValueError("invalid downsampling mode.")

        return latents, not_encoded_latents

    @torch.no_grad()
    def _benchmark_step(self, image, text):
        if self.config.enable_clip_benchmark:
            image = self.clip_preprocess(image).unsqueeze(0)
            text = self.clip_tokenizer([text])

            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return F.cosine_similarity(image_features, text_features).item()
        else:
            return 0

    def render_conditioning_image(self, cameras):
        renderer = self.studio.set_renderer(cameras, self.config.render_size)

        images, fragments = renderer(self.conditioning_mesh)
        return images

    # normalizes image to [-1, 1]
    def normalize_image(self, image):
        return (image * 2 - 1).clamp(-1, 1)
    
    # prepare for classifier-free guidance
    def prepare_conditioning_image_input(self, conditioning_image):
        uncond_image_latents = torch.zeros_like(conditioning_image)
        image_latents = torch.cat(
            [conditioning_image, conditioning_image, uncond_image_latents], dim=0
        )

        return image_latents
    
    def render_frequency_view(self, cameras):
        renderer = self.studio.set_renderer(cameras, self.config.render_size)

        images, _ = renderer(self.frequency_mesh)
        return images

    def fit(self):
        # the only 2 things different here than rgb2x are: textual inversion and conditional attention mask for encoding text prompt

        pbar = tqdm(self.guidance.chosen_ts)

        self.guidance.init_text_embeddings(prompt=self.config.prompt, batch_size=self.config.batch_size)

        for step, chosen_t in enumerate(pbar):

            wandb_log = {}

            Rs, Ts, fovs, ids, image_path = self.studio.sample_cameras(step, self.config.batch_size, self.config.use_random_cameras) 
            cameras = self.studio.set_cameras(self.config.camera_type, Rs, Ts, fovs)

            latents, not_encoded_latents = self.forward(cameras, is_direct=("hashgrid" not in self.config.texture_type))

            if(self.config.use_wandb and self.config.wandb_log_not_encoded_latents):
                wandb_log["clean latents"] = wandb.Image(
                    torchvision.transforms.ToPILImage()(not_encoded_latents[0]).convert("RGB")
                )

            t, noise, noisy_latents = self.guidance.add_noise_to_latents(latents, chosen_t, self.config.batch_size)

            conditioning_image = None
            if(image_path is not None):
                if(self.config.conditioning_image_format == "exr"):
                    full_path = os.path.join(self.config.scene_dir, image_path + ".exr")
                    conditioning_image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                    conditioning_image = cv2.cvtColor(conditioning_image, cv2.COLOR_BGR2RGB)
                    conditioning_image = conditioning_image.astype(np.float32)
                    conditioning_image = torch.from_numpy(conditioning_image).permute(2, 0, 1).unsqueeze(0).to(self.device)

                if(self.config.conditioning_image_format == "png"):
                    full_path = os.path.join(self.config.scene_dir, image_path + ".png")
                    conditioning_image = torchvision.io.read_image(full_path, mode=torchvision.io.ImageReadMode.RGB)
                    conditioning_image = conditioning_image / 255
                    conditioning_image = conditioning_image.unsqueeze(0).to(self.device)


            if(conditioning_image != None):
                conditioning_image_log = conditioning_image
                conditioning_image = self.normalize_image(conditioning_image)
            
                # scaling is also done in encode_latent_texture
                conditioning_image = self.guidance.encode_latent_texture(conditioning_image)
                conditioning_image = self.prepare_conditioning_image_input(conditioning_image)

                # downsize conditioning_image to 768*height (height also needs to be divisible by 8)
                # actually it's 768 * 512

            # compute loss
            self.texture_optimizer.zero_grad()

            x0 = None
            if(self.config.loss_space == "latent"):
                sds_loss = self.guidance.compute_sds_loss(
                    latents, 
                    noisy_latents,
                    noise, 
                    t.to(latents.dtype), 
                    control=conditioning_image
                )

            else:
                sds_loss, x0 = self.guidance.compute_image_space_sds_loss(
                    noisy_latents,
                    not_encoded_latents, 
                    t.to(latents.dtype), 
                    control=conditioning_image
                )

            if hasattr(self, "frequency_mesh") and self.frequency_mesh is not None:
                weighting = self.render_frequency_view(cameras).to(device=self.device, dtype=self.guidance.text_embeddings.dtype)
                weighting = (1 / (2 * weighting.permute(0, 3, 1, 2).clamp(1-6, 1))).clamp(0.1, 10)
                # TODO: log weighting (with clamp(0, 1)) (if too bright, try manual normalization)
                sds_loss = sds_loss * weighting
            
            loss = sds_loss.sum()

            loss.backward()

            '''TODO: uncomment this
            torch.nn.utils.clip_grad_norm_(self._get_texture_parameters(), 1e-1)
                for p in self._get_texture_parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_()
            '''

            self.texture_optimizer.step()
            
            if(self.config.use_wandb):
                wandb.log({
                    "train/sds_loss": loss.item(),
                })
            
            self.avg_loss_sds.append(loss.item())
            
            max_memory_allocated = torch.cuda.max_memory_allocated()
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {t.item()}, GPU: {max_memory_allocated / 1024**3:.2f} GB')

            if step % self.config.local_log_steps == 0 and self.config.use_local_log:
                # save texture field
                checkpoint = {
                    "render_func": self.studio.render_func.state_dict()
                }

                if "hashgrid" in self.config.texture_type:
                    checkpoint["texture"] = self.texture_mesh.texture.state_dict()
                else:
                    checkpoint["texture"] = self.texture_mesh.texture

                if self.config.enable_anchor_embedding: 
                    checkpoint["anchor_func"] = self.studio.anchor_func.state_dict()

                ''' TODO: do we need checkpoint logging?
                torch.save(
                    checkpoint,
                    os.path.join(self.log_dir, "checkpoint_{}.pth".format(step))
                )
                '''

                # visualize
                if self.config.show_original_texture:
                    if self.config.texture_type == "latent":
                        decoded_texture = self.guidance.decode_latent_texture(self.texture_mesh.texture.permute(0, 3, 1, 2))
                        decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0]).convert("RGB")
                        decoded_texture.save(os.path.join(self.config.log_dir, "texture_{}.png".format(step)))
                    elif self.config.texture_type == "rgb":
                        decoded_texture = (self.texture_mesh.texture / 2 + 0.5).clamp(0, 1)
                        decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0].permute(2, 0, 1)).convert("RGB")
                        decoded_texture.save(os.path.join(self.config.log_dir, "texture_{}.png".format(step)))
                    else:
                        self.inference(self.config.log_dir, step, self.config.texture_size)
            
            if step % self.config.wandb_log_steps == 0 and self.config.use_wandb:
                wandb_log["train/avg_loss"] = np.mean(self.avg_loss_sds)
                wandb_log["time step"] = chosen_t
                
                if self.config.wandb_log_denoised_latents:
                    renderings = []
                    for view_id in range(self.config.log_latents_views):
                        Rs, Ts, fovs, _, _ = self.studio.sample_cameras(view_id, 1, random_cameras=False, inference=False)
                        cameras = self.studio.set_cameras(self.config.camera_type, Rs, Ts, fovs)

                        if self.config.texture_type == "latent":
                                with torch.no_grad():
                                    latents, _ = self.forward(cameras, False, True, is_direct=("hashgrid" not in self.config.texture_type))
                                    latents = self.guidance.decode_latent_texture(latents)
                        else:
                            with torch.no_grad():
                                latents, _ = self.forward(cameras, False, False, is_direct=("hashgrid" not in self.config.texture_type))
                                latents = (latents / 2 + 0.5).clamp(0, 1)
                        
                        if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                            latents = latents[:, 0:1].repeat(1, 3, 1, 1)
                        latents_image = torchvision.transforms.ToPILImage()(latents[0]).convert("RGB").resize((self.config.decode_size, self.config.decode_size))
                        renderings.append(wandb.Image(latents_image))

                    wandb_log["denoised latents"] = renderings

                if self.config.wandb_log_conditioning_image:
                    conditioning_image_log = torchvision.transforms.ToPILImage()(conditioning_image_log[0])
                    conditioning_image_log = conditioning_image_log.convert("RGB")
                    conditioning_image_log = conditioning_image_log.resize((self.config.decode_size, self.config.decode_size))
                    wandb_conditioning_rendering = wandb.Image(conditioning_image_log)
                    wandb_log["conditioning image"] = wandb_conditioning_rendering

                if self.config.wandb_log_pred_original_sample and x0 is not None:
                    x0_log = x0
                    if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                            x0_log = x0_log[:, 0:1].repeat(1, 3, 1, 1)
                    x0_log = torchvision.transforms.ToPILImage()(x0_log[0]).convert("RGB")
                    wandb_x0_rendering = wandb.Image(x0_log)
                    wandb_log["original sample predicate"] = wandb_x0_rendering

                if self.config.wandb_log_noisy_latents:
                    with torch.no_grad():
                        noisy_latents_log = noisy_latents
                        if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                            noisy_latents_log = noisy_latents_log[:, 0:1].repeat(1, 3, 1, 1)
                        noisy_latents_log = 1 / self.guidance.vae.config.scaling_factor * noisy_latents_log
                        noisy_latents_log = self.guidance.vae.decode(noisy_latents_log.contiguous()).sample # B, 3, H, W
                        noisy_latents_log = (noisy_latents_log / 2 + 0.5).clamp(0, 1)
                        noisy_latents_log = torchvision.transforms.ToPILImage()(noisy_latents_log[0]).convert("RGB")
                        wandb_noisy_latents_rendering = wandb.Image(noisy_latents_log)
                        wandb_log["noisy latents"] = wandb_noisy_latents_rendering

                if self.config.wandb_log_texture:
                    decoded_texture = self.texture_mesh.texture.permute(0, 3, 1, 2)
                    if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                            decoded_texture = decoded_texture[:, 0:1].repeat(1, 3, 1, 1)
                    decoded_texture = (decoded_texture / 2 + 0.5).clamp(0, 1)
                    decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0]).convert("RGB")
                    decoded_texture = wandb.Image(decoded_texture)
                    wandb_log["texture"] = decoded_texture

                wandb.log(wandb_log)


        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.render_all_views(timestamp)

        self.save_texture(timestamp)

        wandb.finish()

    def render_all_views(self, timestamp):
        directory = os.path.join(self.config.log_dir, "kitchen_hq", timestamp, "{}_results".format(self.config.aov))
        if not os.path.exists(directory):
            os.makedirs(directory)

        cameras_count = self.studio.num_cameras
        
        for camera_id in range(cameras_count):
            Rs, Ts, fovs, _, image_path = self.studio.sample_cameras(camera_id, self.config.batch_size, random_cameras=False)            
            cameras = self.studio.set_cameras(self.config.camera_type, Rs, Ts, fovs)

            _, features = self.forward(cameras, inference=True, downsample=False, is_direct=True)
            if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                features = features[:, 0:1].repeat(1, 3, 1, 1)

            image = torchvision.transforms.ToPILImage()(features.squeeze(0))
            path = os.path.join(directory, "{}.png".format(image_path.split("/")[-1]))
            image.save(path)

    def save_texture(self, timestamp):
        directory = os.path.join(self.config.log_dir, "kitchen_hq", timestamp, "{}_results".format(self.config.aov))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        decoded_texture = self.texture_mesh.texture.permute(0, 3, 1, 2)
        if(self.config.aov == "roughness" or self.config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
                decoded_texture = decoded_texture[:, 0:1].repeat(1, 3, 1, 1)
        decoded_texture = (decoded_texture / 2 + 0.5).clamp(0, 1)
        decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0]).convert("RGB")
        decoded_texture.save(os.path.join(directory, "{}_texture.png".format(self.config.aov)))
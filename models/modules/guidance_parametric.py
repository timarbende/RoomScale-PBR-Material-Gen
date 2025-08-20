import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from diffusers import DDIMScheduler
from models.pipeline.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

# customized
import sys
sys.path.append("./models")
from models.utils.lora import extract_lora_diffusers

class Guidance(nn.Module):
    def __init__(self, 
        config,
        device
    ): 
        
        super().__init__()
        
        self.config = config
        self.device = device

        self.aov= config.aov
        prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        }
        self.prompt=prompts[self.aov]
        self.n_prompt = config.n_prompt
        
        self.weights_dtype = torch.float16 if self.config.enable_half_precision else torch.float32

        self._init_guidance()

    def _init_guidance(self):
        self._init_backbone()
        self._init_t_schedule()

    def _init_backbone(self):
        mat_est_pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
            "zheng95z/rgb-to-x",
            torch_dtype=torch.float32,
            cache_dir=self.config.cache_dir,
        ).to(self.device)

        if self.config.enable_memory_efficient_attention:
            print("=> Enable memory efficient attention.")
            mat_est_pipe.enable_xformers_memory_efficient_attention()

        # pretrained diffusion model
        self.tokenizer = mat_est_pipe.tokenizer
        self.text_encoder = mat_est_pipe.text_encoder
        self.vae = mat_est_pipe.vae
        self.unet = mat_est_pipe.unet.to(self.weights_dtype)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.scheduler = mat_est_pipe.scheduler
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.num_train_timesteps = len(self.scheduler.betas)
        self.scheduler.set_timesteps(self.num_train_timesteps)

        # loss weights
        self.loss_weights = self._init_loss_weights(self.scheduler.betas)

        self.avg_loss_vsd = []
        self.avg_loss_phi = []
        self.avg_loss_rgb = []

        max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"=> Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")

    def _init_loss_weights(self, betas):    
        num_train_timesteps = len(betas)
        betas = torch.tensor(betas).to(torch.float32) if not torch.is_tensor(betas) else betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            
        weights = []
        for i in range(num_train_timesteps):
            weights.append(sqrt_1m_alphas_cumprod[i]**2)
            
        return weights
    
    def _init_t_schedule(self, t_start=0.02, t_end=0.98):
        # Create a list of time steps from 0 to num_train_timesteps
        ts = list(range(self.num_train_timesteps))
        # set ts to U[0.02,0.98] as least
        t_start = int(t_start * self.num_train_timesteps)
        t_end = int(t_end * self.num_train_timesteps)
        ts = ts[t_start:t_end]

        # If the scheduling strategy is "random", choose args.num_steps random time steps without replacement
        if self.config.t_schedule == "random":
            chosen_ts = np.random.choice(ts, self.config.num_steps, replace=True)

        # If the scheduling strategy is "t_stages", the total number of time steps are divided into several stages.
        # In each stage, a decreasing portion of the total time steps is considered for selection.
        # For each stage, time steps are randomly selected with replacement from the respective portion.
        # The final list of chosen time steps is a concatenation of the time steps selected in all stages.
        # Note: The total number of time steps should be evenly divisible by the number of stages.
        elif "t_stages" in self.config.t_schedule:
            # Parse the number of stages from the scheduling strategy string
            num_stages = int(self.config.t_schedule[8:]) if len(self.config.t_schedule[8:]) > 0 else 2
            chosen_ts = []
            for i in range(num_stages):
                # Define the portion of ts to be considered in this stage
                portion = ts[:int((num_stages-i)*len(ts)//num_stages)]
                selected_ts = np.random.choice(portion, self.config.num_steps//num_stages, replace=True).tolist()
                chosen_ts += selected_ts
        
        elif "anneal" in self.config.t_schedule:
            print("=> time step annealing after {} steps".format(self.config.num_anneal_steps))

            ts_before_anneal = np.random.choice(ts, self.config.num_anneal_steps, replace=True).tolist()
            ts_after_anneal = np.random.choice(ts[:len(ts)//2], self.config.num_steps-self.config.num_anneal_steps, replace=True).tolist()
            chosen_ts = ts_before_anneal + ts_after_anneal
        
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.config.t_schedule}")

        # Return the list of chosen time steps
        self.chosen_ts = chosen_ts

    def init_text_embeddings(self, batch_size):
        ### get text embedding
        text_inputs = self.tokenizer(
            self.prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        self._check_for_removed_text(text_input_ids)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]

        text_embeddings = text_embeddings.to(dtype=self.text_encoder.dtype, device=self.device)

        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [self.n_prompt], 
            padding="max_length", 
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))

        uncond_embeddings = uncond_embeddings[0].to(
                dtype=self.text_encoder.dtype, device=self.device
            )

        self.text_embeddings = torch.cat([text_embeddings, uncond_embeddings, uncond_embeddings])


    def _check_for_removed_text(self, text_input_ids):
        untruncated_ids = self.tokenizer(
                self.prompt, padding="longest", return_tensors="pt"
            ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
    
    @torch.no_grad()
    def decode_latent_texture(self, inputs, use_patches=False):
        outputs = 1 / self.vae.config.scaling_factor * inputs

        if use_patches:
            assert self.config.latent_texture_size % self.config.decode_texture_size == 0
            batch_size = inputs.shape[0]
            num_iter_x = self.config.latent_texture_size // self.config.decode_texture_size
            num_iter_y = self.config.latent_texture_size // self.config.decode_texture_size
            patch_stride = self.config.decode_texture_size
            decoded_stride = self.config.decode_texture_size * 8
            decoded_size = self.config.latent_texture_size * 8
            decoded_texture = torch.zeros(batch_size, 3, decoded_size, decoded_size).to(self.device)

            for x in range(num_iter_x):
                for y in range(num_iter_y):
                    patch = outputs[:, :, x*patch_stride:(x+1)*patch_stride, y*patch_stride:(y+1)*patch_stride]
                    patch = self.vae.decode(patch.contiguous()).sample # B, 3, H, W

                    decoded_texture[:, :, x*decoded_stride:(x+1)*decoded_stride, y*decoded_stride:(y+1)*decoded_stride] = patch
        
            outputs = (decoded_texture / 2 + 0.5).clamp(0, 1)

        else:
            outputs = self.vae.decode(outputs.contiguous()).sample # B, 3, H, W
            outputs = (outputs / 2 + 0.5).clamp(0, 1)

        return outputs
    
    def encode_latent_texture(self, inputs, deterministic=False):
        inputs = inputs.clamp(-1, 1)
        
        h = self.vae.encoder(inputs)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.zeros_like(mean) if deterministic else torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(mean)
        
        return self.vae.config.scaling_factor * sample
    
    def prepare_one_latent(self, latents, t):
        noise = torch.randn_like(latents).to(self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        clean_latents = self.scheduler.step(noise, t, noisy_latents).pred_original_sample

        return noise, noisy_latents, clean_latents

    def prepare_latents(self, latents, t, batch_size):
        t = torch.tensor([t]).to(self.device)
        noise, noisy_latents, clean_latents = self.prepare_one_latent(latents, t)

        return t, noise, noisy_latents, clean_latents
    
    def predict_noise(self, unet, noisy_latents, t, cross_attention_kwargs, guidance_scale, image_guidance_scale, control=None):

        down_block_res_samples, mid_block_res_sample = None, None

        # only kept CFG branch of if-else
        latent_model_input = (
                torch.cat([noisy_latents] * 3)                
            )

        scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
        if control is not None:
            scaled_latent_model_input = torch.cat(
                    [scaled_latent_model_input, control], dim=1
                )

        noise_pred = unet(
            scaled_latent_model_input.to(self.weights_dtype), 
            t, 
            encoder_hidden_states=self.text_embeddings.to(self.weights_dtype), 
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        ).sample.to(torch.float32)

        # perform guidance  #TODO: check if we actually do guidance here (but I guess we always do?)
        (
            noise_pred_text,
            noise_pred_image,
            noise_pred_uncond,
        ) = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )

        '''TODO: do we need this? not yet
        if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )
        '''

        return noise_pred

    def compute_sds_loss(self, latents, noisy_latents, noise, t, control=None):
        with torch.no_grad():
            noise_pred = self.predict_noise(
                self.unet, 
                noisy_latents, 
                t,
                cross_attention_kwargs={},
                guidance_scale=self.config.guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                control=control
            )

        # TODO get x0 (completely denoised image: schedulerből a pred_original_sample)
        # x0 = decode(ddim_scale(noisy_latent - noise_pred)), de ezt megkaphatjuk a pred_original_sample-l
        # TODO: image-space loss: grad = z0 - x0

        grad = self.config.grad_scale * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        grad *= self.loss_weights[int(t)]
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad

        # TODO: in the long run we will probably need to use image-space loss (decode into rgb and compute loss there)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean")

        return loss
    
    def compute_image_space_sds_loss(self, noisy_latents, not_encoded_latents, t, control=None):
        with torch.no_grad():
            noise_pred = self.predict_noise(
                self.unet, 
                noisy_latents, 
                t,
                cross_attention_kwargs={},
                guidance_scale=self.config.guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                control=control
            )

            x0 = self.scheduler.step(noise_pred, int(t), noisy_latents).pred_original_sample
            x0 = self.vae.decode(x0).sample
            x0 = (x0 / 2 + 0.5).clamp(0, 1)

        loss = 0.5 * F.mse_loss(not_encoded_latents, x0, reduction="mean")

        return loss
    
    def encode_image(self, image):
        # this line only works for singular batch. for batch with multiple elements check rgb2x prepare_image_latents
        image_latents = self.vae.encode(image).latent_dist.mode()
        image_latents = torch.cat([image_latents], dim=0)
        
        return image_latents
    
    def scale_conditioning_image(self, image):
        return image * self.vae.config.scaling_factor

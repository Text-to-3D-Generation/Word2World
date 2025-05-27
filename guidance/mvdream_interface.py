import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mvdream.camera_utils import normalize_camera
from mvdream.model_zoo import build_model
from diffusers import DDIMScheduler

class MVDream(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.dtype = torch.float16
        self.model = build_model('sd-v2.1-base-4view', ckpt_path=None).eval().to(self.device)
        self.model.half()  # Convert all model weights to FP16.
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps*0.02)
        self.max_step = int(self.num_train_timesteps*0.098)
        self.embeddings = {}
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype)

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=["ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"]):
        self.embeddings['pos'] = self.model.get_learned_conditioning(prompt).to(self.device).to(self.dtype).repeat(4, 1, 1)
        self.embeddings['neg'] = self.model.get_learned_conditioning(negative_prompt).to(self.device).to(self.dtype).repeat(4, 1, 1)

    def train_step(self,pred_rgb,camera,step_ratio):
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
        imgs = 2 * pred_rgb_256 - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
        t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        camera = camera[:, [0, 2, 1, 3]]
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)
        camera = camera.to(self.dtype)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1),self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}
        with torch.no_grad():
            noise = torch.randn_like(latents).to(self.dtype)
            latents_noisy = self.model.q_sample(latents, t, noise)
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2).float().to(self.device).to(self.dtype)
            with autocast(dtype=self.dtype):
                noise_pred = self.model.apply_model(latent_model_input, tt, context)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 100*(noise_pred_pos - noise_pred_uncond)

        grad = (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        return loss





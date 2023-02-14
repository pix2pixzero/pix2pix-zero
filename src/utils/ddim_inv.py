import sys
import numpy as np
import torch
import torch.nn.functional as F
from random import randrange
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class DDIMInversion(BasePipeline):

    def auto_corr_loss(self, x, random_shift=True):
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inversion_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        img=None, # the input image as a PIL image
        torch_dtype=torch.float32,

        # inversion regularization parameters
        lambda_ac: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_ac_rolls: int = 5,
    ):
        
        # 0. modify the unet to be useful :D
        self.unet = prep_unet(self.unet)

        # set the scheduler to be the Inverse DDIM scheduler
        # self.scheduler = MyDDIMScheduler.from_config(self.scheduler.config)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Encode the input image with the first stage model
        x0 = np.array(img)/255
        x0 = torch.from_numpy(x0).type(torch_dtype).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
        x0 = (x0 - 0.5) * 2.
        with torch.no_grad():
            x0_enc = self.vae.encode(x0).latent_dist.sample().to(device, torch_dtype)
        latents = x0_enc = 0.18215 * x0_enc

        # Decode and return the image
        with torch.no_grad():
            x0_dec = self.decode_latents(x0_enc.detach())
        image_x0_dec = self.numpy_to_pil(x0_dec)

        with torch.no_grad():
            prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt).to(device)
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)

        # Do the inversion
        num_warmup_steps = len(timesteps) - num_inversion_steps * self.scheduler.order # should be 0?
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(timesteps.flip(0)[1:-1]):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # regularization of the noise prediction
                e_t = noise_pred
                for _outer in range(num_reg_steps):
                    if lambda_ac>0:
                        for _inner in range(num_ac_rolls):
                            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                            l_ac = self.auto_corr_loss(_var)
                            l_ac.backward()
                            _grad = _var.grad.detach()/num_ac_rolls
                            e_t = e_t - lambda_ac*_grad
                    if lambda_kl>0:
                        _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                        l_kld = self.kl_divergence(_var)
                        l_kld.backward()
                        _grad = _var.grad.detach()
                        e_t = e_t - lambda_kl*_grad
                    e_t = e_t.detach()
                noise_pred = e_t

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, reverse=True, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        

        x_inv = latents.detach().clone()
        # reconstruct the image

        # 8. Post-processing
        image = self.decode_latents(latents.detach())
        image = self.numpy_to_pil(image)
        return x_inv, image, image_x0_dec

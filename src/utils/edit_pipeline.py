import pdb, sys

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EditingPipeline(BasePipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        x_in=None,
        only_sample=False, # only perform sampling, and no editing

    ):

        x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 0. modify the unet to be useful :D
        self.unet = prep_unet(self.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)
        # 3. Encode input prompt = 2x77x1024
        prompt_embeds = self._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "CrossAttention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))

        if only_sample:
            return image_rec


        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        prompt_embeds_edit[1:2] += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                noise_pred = self.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),cross_attention_kwargs=cross_attention_kwargs,).sample

                loss = 0.0
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "CrossAttention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                
                latents = x_in.detach().chunk(2)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        # 8. Post-processing
        image = self.decode_latents(latents.detach())

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image_edit = self.numpy_to_pil(image)


        return image_rec, image_edit

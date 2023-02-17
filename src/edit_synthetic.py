import os, pdb

import argparse
import numpy as np
import torch
import requests
from PIL import Image

from diffusers import DDIMScheduler
from utils.edit_directions import construct_direction
from utils.edit_pipeline import EditingPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_str', type=str, required=True)
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--xa_guidance', default=0.15, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1,4,64,64), device=device)

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    rec_pil, edit_pil = pipe(args.prompt_str, 
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        edit_dir=construct_direction(args.task_name),
        guidance_amount=args.xa_guidance,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt="" # use the empty string for the negative prompt
    )
    
    edit_pil[0].save(os.path.join(args.results_folder, f"edit.png"))
    rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction.png"))

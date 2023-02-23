import os
import sys
import shutil

import torch
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel
from lavis.models import load_model_and_preprocess
from diffusers import DDIMScheduler

sys.path.insert(0, "src")
from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler
from utils.edit_directions import construct_direction
from utils.edit_pipeline import EditingPipeline


MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_CACHE = "diffusers-cache"


class ModelOutput(BaseModel):
    reconstructed_image: Path
    caption_input_image: str
    edited_image: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        # load the BLIP model
        self.model_blip, self.vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=torch.device("cuda"),
        )

    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        task: str = Input(
            description="Describe how to edit the image", default="cat2dog"
        ),
        xa_guidance: float = Input(
            description="",
            default=0.1,
        ),
        negative_guidance_scale: float = Input(
            description="Number of images to output.",
            default=5.0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        use_float_16: bool = Input(
            description="Choose half precision if set to True", default=True
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        torch_dtype = torch.float16 if use_float_16 else torch.float32

        inversion_pipe = DDIMInversion.from_pretrained(
            MODEL_ID,
            cache_dir=f"{MODEL_CACHE}/inversion",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to("cuda")

        inversion_pipe.scheduler = DDIMInverseScheduler.from_config(
            inversion_pipe.scheduler.config
        )

        enditing_pipe = EditingPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=f"{MODEL_CACHE}/edit",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to("cuda")
        enditing_pipe.scheduler = DDIMScheduler.from_config(
            enditing_pipe.scheduler.config
        )

        img = Image.open(str(image)).resize((512, 512), Image.Resampling.LANCZOS)
        # generate the caption
        _image = self.vis_processors["eval"](img).unsqueeze(0).cuda()
        prompt_str = self.model_blip.generate({"image": _image})[0]
        x_inv, x_inv_image, x_dec_img = inversion_pipe(
            prompt_str,
            guidance_scale=1,
            num_inversion_steps=num_inference_steps,
            img=img,
            torch_dtype=torch_dtype,
        )

        print(f"Image caption generated with BLIP model: {prompt_str}")

        # save the inversion
        inversion_path = "cog_inversion_path"
        if os.path.exists(inversion_path):
            shutil.rmtree(inversion_path)
        os.makedirs(inversion_path)

        torch.save(x_inv[0], os.path.join(inversion_path, "inversion.pt"))

        rec_pil, edit_pil = enditing_pipe(
            prompt_str,
            num_inference_steps=num_inference_steps,
            x_in=torch.load(os.path.join(inversion_path, "inversion.pt")).unsqueeze(0),
            edit_dir=construct_direction(task),
            guidance_amount=xa_guidance,
            guidance_scale=negative_guidance_scale,
            negative_prompt=prompt_str,  # use the unedited prompt for the negative prompt
        )

        reconstructed_image = "/tmp/reconstruction.png"
        edited_image = "/tmp/edit.png"
        edit_pil[0].save(edited_image)
        rec_pil[0].save(reconstructed_image)

        return ModelOutput(
            reconstructed_image=Path(reconstructed_image),
            caption_input_image=prompt_str,
            edited_image=Path(edited_image),
        )

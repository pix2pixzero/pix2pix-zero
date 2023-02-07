# pix2pix-zero [diffusers]

### [website](https://pix2pixzero.github.io/)

## Code and Demo coming soon


<br>
<div class="gif">
<p align="center">
<img src='assets/main.gif' align="center">
</p>
</div>


We propose pix2pix-zero, a diffusion-based image-to-image approach that allows users to specify the edit direction on-the-fly (e.g., cat to dog). Our method can directly use pre-trained [Stable Diffusion](https://github.com/CompVis/stable-diffusion), for editing real and synthetic images while preserving the input image's structure. Our method is training-free and prompt-free, as it requires neither manual text prompting for each input image nor costly fine-tuning for each task.


## Results
All our results are based on [stable-diffusion-v1-4](https://github.com/CompVis/stable-diffusion) model. Please the website for more results.




<div>
<p align="center">
<img src='assets/results_teaser.jpg' align="center" width=800px>
</p>
</div>

## Real Image Editing
<div>
<p align="center">
<img src='assets/results_real.jpg' align="center" width=800px>
</p>
</div>

## Synthetic Image Editing
<div>
<p align="center">
<img src='assets/results_syn.jpg' align="center" width=800px>
</p>
</div>

## Method Details

Given an input image, we first generate text captions using [BLIP](https://github.com/salesforce/LAVIS) and apply regularized DDIM inversion to obtain our inverted noise map.
Then, we obtain reference cross-attention maps that correspoind to the structure of the input image by denoising, guided with the CLIP embeddings 
of our generated text (c). Next, we denoise with edited text embeddings, while enforcing a loss to match current cross-attention maps with the 
reference cross-attention maps.

<div>
<p align="center">
<img src='assets/method.jpeg' align="center" width=900>
</p>
</div>



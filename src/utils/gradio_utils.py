import os, sys, time, re, pdb
import torch, torchvision
import numpy
from PIL import Image
import hashlib
from tqdm import tqdm
import openai
# from utils.direction_utils import *
import gradio as gr
from diffusers import DDIMScheduler

p = "src/utils"
if p not in sys.path:
    sys.path.append(p)
from huggingface_utils import *
from edit_directions import construct_direction
from edit_pipeline import EditingPipeline
from ddim_inv import DDIMInversion
from scheduler import DDIMInverseScheduler
from lavis.models import load_model_and_preprocess
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, BloomForCausalLM



"""
    Load sentence embeddings for a list of sentences.

    Args:
        l_sentences (list of str): List of sentences to embed.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the sentences.
        text_encoder (PreTrainedModel): Text encoder for generating embeddings.
        device (str): Device to use for processing (default: "cuda").

    Returns:
        Tensor: Mean of the sentence embeddings for all input sentences.
"""
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
        l_embeddings = []
        for sent in tqdm(l_sentences):
            text_inputs = tokenizer(
                    sent,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concatenate(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)


"""
    Generate a sample image using the provided prompt and parameters.

    Args:
        prompt (str): Prompt text for generating the image.
        seed (int): Random seed to use for generating the noise map.
        negative_scale (float): Scale of the negative guidance to use.
        num_ddim (int): Number of diffusion steps to use.

    Returns:
        tuple: A tuple containing the generated PIL image and the filename of the inverse noise map.
"""
def launch_generate_sample(prompt, seed, negative_scale, num_ddim):
    os.makedirs("tmp", exist_ok=True)
    # do the editing
    edit_pipe = EditingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
    edit_pipe.scheduler = DDIMScheduler.from_config(edit_pipe.scheduler.config)
    # set the random seed and sample the input noise map
    torch.cuda.manual_seed(int(seed))
    z = torch.randn((1,4,64,64), device="cuda")
    z_hashname = hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
    z_inv_fname = f"tmp/{z_hashname}_ddim_{num_ddim}_inv.pt"
    torch.save(z, z_inv_fname)
    rec_pil = edit_pipe(prompt, 
        num_inference_steps=num_ddim, x_in=z,
        only_sample=True, # this flag will only generate the sampled image, not the edited image
        guidance_scale=negative_scale,
        negative_prompt="" # use the empty string for the negative prompt
    )
    del edit_pipe
    torch.cuda.empty_cache()
    return rec_pil[0], z_inv_fname


"""
    Clean a list of sentences by removing digits and special characters.

    Args:
        ls (list of str): List of sentences to clean.

    Returns:
        list of str: List of cleaned sentences.
"""
def clean_l_sentences(ls):
    s = [re.sub('\d', '', x) for x in ls]
    s = [x.replace(".","").replace("-","").replace(")","").strip() for x in s]
    return s


"""
    Use the OpenAI API to generate sentences related to the specified word.

    Args:
        task_type (str): Type of task to perform ("object" or "style").
        word (str): Word to generate sentences for.
        num (int): Maximum number of sentences to generate (default: 100).

    Returns:
        list of str: List of generated sentences.
"""
def gpt3_compute_word2sentences(task_type, word, num=100):
    l_sentences = [] 
    if task_type=="object":
        template_prompt = f"Provide many captions for images containing {word}."
    elif task_type=="style":
        template_prompt = f"Provide many captions for images that are in the {word} style."
    while True:
        ret = openai.Completion.create(
            model="text-davinci-002",
            prompt=template_prompt,
            max_tokens=1000,
            temperature=1.0)
        raw_return = ret.choices[0].text
        for line in raw_return.split("\n"):
            line = line.strip()
            if len(line)>10:
                skip=False 
                for subword in word.split(" "):
                    if subword not in line: skip=True
                if not skip: l_sentences.append(line)
                else:
                    l_sentences.append(line+f", {word}")
        time.sleep(0.05)
        print(len(l_sentences))
        if len(l_sentences)>=num:
            break
    l_sentences = clean_l_sentences(l_sentences)
    return l_sentences


"""
    Use the Flan T5-XL model to generate sentences related to the specified word.

    Args:
        word (str): Word to generate sentences for.
        num (int): Maximum number of sentences to generate (default: 100).

    Returns:
        list of str: List of generated sentences.
"""
def flant5xl_compute_word2sentences(word, num=100):
    text_input = f"Provide a caption for images containing a {word}. The captions should be in English and should be no longer than 150 characters. The caption must contain the word {word}. Caption:"
    l_sentences = []
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids.to("cuda")
    input_length = input_ids.shape[1]
    while True:
        try:
            outputs = model.generate(input_ids,temperature=0.95, num_return_sequences=16, do_sample=True, max_length=128, min_length=15, eta_cutoff=1e-5)
            # output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except:
            continue
        for line in output:
            line = line.strip()
            # print(line)
            l_sentences.append(line)
        print(len(l_sentences))
        if len(l_sentences)>=num:
            break
    l_sentences = clean_l_sentences(l_sentences)
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return l_sentences


"""
    Use the BLOOMZ model to generate sentences related to the specified word.

    Args:
        word (str): Word to generate sentences for.
        num (int): Maximum number of sentences to generate (default: 100).

    Returns:
        list of str: List of generated sentences.
"""
def bloomz_compute_sentences(word, num=100):
    l_sentences = []
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
    model = BloomForCausalLM.from_pretrained("bigscience/bloomz-7b1", device_map="auto", torch_dtype=torch.float16)
    text_input = f"Provide a caption for images containing a {word}. The captions should be in English and should be no longer than 150 characters. The caption must contain the word {word}. Caption:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    input_length = input_ids.shape[1]
    t = 0.95
    eta = 1e-5
    min_length = 15

    while True:
        try:
            outputs = model.generate(input_ids,temperature=t, num_return_sequences=16, do_sample=True, max_length=128, min_length=min_length, eta_cutoff=eta)
            output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        except:
            continue
        for line in output:
            line = line.strip()
            skip=False 
            for subword in word.split(" "):
                if subword not in line: skip=True
            print(line)
            if not skip: l_sentences.append(line)
            else: l_sentences.append(line+f", {word}")
            
        print(len(l_sentences))
        if len(l_sentences)>=num:
            break
    l_sentences = clean_l_sentences(l_sentences)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return l_sentences



def generate_image_prompts_with_templates(word):
    prompts = []
    adjectives = ['majestic', 'cute', 'colorful', 'ferocious', 'elegant', 'graceful', 'slimy', 'adorable', 'scary', 'fuzzy', 'tiny', 'gigantic', 'brave', 'fierce', 'mysterious', 'curious', 'fascinating', 'charming', 'gleaming', 'rare']
    verbs = ['strolling', 'jumping', 'lounging', 'flying', 'sleeping', 'eating', 'playing', 'working', 'gazing', 'standing']
    adverbs = ['gracefully', 'playfully', 'elegantly', 'fiercely', 'curiously', 'fascinatingly', 'charmingly', 'gently', 'slowly', 'quickly', 'awkwardly', 'carelessly', 'cautiously', 'innocently', 'powerfully', 'grumpily', 'mysteriously']
    backgrounds = ['a sunny beach', 'a bustling city', 'a quiet forest', 'a cozy living room', 'a futuristic space station', 'a medieval castle', 'an enchanted forest', 'a misty graveyard', 'a snowy mountain peak', 'a crowded market']

    sentence_structures = {
        "subject verb background": lambda word, bg, verb, adj, adv: f"A {word} {verb} {bg}.",
        "background subject verb": lambda word, bg, verb, adj, adv: f"{bg}, a {word} is {verb}.",
        "adjective subject verb background": lambda word, bg, verb, adj, adv: f"A {adj} {word} is {verb} {bg}.",
        "subject verb adverb background": lambda word, bg, verb, adj, adv: f"A {word} is {verb} {adv} {bg}.",
        "adverb subject verb background": lambda word, bg, verb, adj, adv: f"{adv.capitalize()}, a {word} is {verb} {bg}.",
        "background adjective subject verb": lambda word, bg, verb, adj, adv: f"{bg}, there is a {adj} {word} {verb}.",
        "subject verb adjective background": lambda word, bg, verb, adj, adv: f"A {word} {verb} {adj} {bg}.",
        "adjective subject verb": lambda word, bg, verb, adj, adv: f"A {adj} {word} is {verb}.",
        "subject adjective verb background": lambda word, bg, verb, adj, adv: f"A {word} is {adj} and {verb} {bg}.",
    }

    sentences = []
    for bg in backgrounds:
        for verb in verbs:
            for adj in adjectives:
                adv = random.choice(adverbs)
                sentence = f"A {adv} {adj} {word} {verb} on {bg}."
                sentence_structure = random.choice(list(sentence_structures.keys()))
                sentence = sentence_structures[sentence_structure](word, bg, verb, adj, adv)
                sentences.append(sentence)
    return sentences



"""
    Create a custom directory of image embeddings for a specified description using various sentence generation methods.

    Args:
        description (str): Description of the images.
        sent_type (str): Type of sentence generation method to use ("fixed-template", "GPT3", "flan-t5-xl", "BLOOMZ-7B", or "custom sentences").
        api_key (str): OpenAI API key (required if using GPT-3).
        org_key (str): OpenAI organization key (required if using GPT-3).
        l_custom_sentences (str): Custom sentences to use (required if using "custom sentences" sentence generation method).

    Returns:
        torch.Tensor: Tensor of image embeddings.
"""
def make_custom_dir(description, sent_type, api_key, org_key, l_custom_sentences):
    if sent_type=="fixed-template":
        l_sentences = generate_image_prompts_with_templates(description)
    elif "GPT3" in sent_type:
        import openai
        openai.organization = org_key
        openai.api_key = api_key
        _=openai.Model.retrieve("text-davinci-002")
        l_sentences = gpt3_compute_word2sentences("object", description, num=1000)
    
    elif "flan-t5-xl" in sent_type:
        l_sentences = flant5xl_compute_word2sentences(description, num=1000)
        # save the sentences to file
        with open(f"tmp/flant5xl_sentences_{description}.txt", "w") as f:
            for line in l_sentences:
                f.write(line+"\n")
    elif "BLOOMZ-7B" in sent_type:
        l_sentences = bloomz_compute_sentences(description, num=1000)
        # save the sentences to file
        with open(f"tmp/bloomz_sentences_{description}.txt", "w") as f:
            for line in l_sentences:
                f.write(line+"\n")
    
    elif sent_type=="custom sentences":
        l_sentences = l_custom_sentences.split("\n")
        print(f"length of new sentence is {len(l_sentences)}")

    pipe = EditingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
    emb = load_sentence_embeddings(l_sentences, pipe.tokenizer, pipe.text_encoder, device="cuda")
    del pipe
    torch.cuda.empty_cache()
    return emb


"""
    Edit an image using a text prompt and custom directions.

    Args:
        img_in_real (PIL.Image.Image or None): The input real image to be edited. If None, then `img_in_synth` must not be None.
        img_in_synth (PIL.Image.Image or None): The input synthetic image to be edited. If None, then `img_in_real` must not be None.
        src (str): The name of the source direction to use for the edit.
        src_custom (str): The custom description to use for the source direction, if `src` is "make your own!".
        dest (str): The name of the destination direction to use for the edit.
        dest_custom (str): The custom description to use for the destination direction, if `dest` is "make your own!".
        num_ddim (int): The number of diffusion steps to use for the edit.
        xa_guidance (float): The amount of cross-attention guidance to use for the edit.
        edit_mul (float): The amount to scale the direction vector by for the edit.
        fpath_z_gen (str): The file path to the input noise map for the synthetic image.
        gen_prompt (str): The prompt to use for the synthetic image.
        sent_type_src (str): The type of sentence generation to use for the source direction.
        sent_type_dest (str): The type of sentence generation to use for the destination direction.
        api_key (str): The API key to use for OpenAI API calls, if necessary.
        org_key (str): The organization ID to use for OpenAI API calls, if necessary.
        custom_sentences_src (str): The custom sentences to use for the source direction, if `sent_type_src` is "custom sentences".
        custom_sentences_dest (str): The custom sentences to use for the destination direction, if `sent_type_dest` is "custom sentences".

    Returns:
        The edited PIL image resulting from the edit.
"""
def launch_main(img_in_real, img_in_synth, src, src_custom, dest, dest_custom, num_ddim, xa_guidance, edit_mul, fpath_z_gen, gen_prompt, sent_type_src, sent_type_dest, api_key, org_key, custom_sentences_src, custom_sentences_dest):
    d_name2desc = hf_get_all_directions_names()
    d_desc2name = {v:k for k,v in d_name2desc.items()}
    os.makedirs("tmp", exist_ok=True)

    # generate custom direction first
    if src=="make your own!":
        outf_name = f"tmp/template_emb_{src_custom}_{sent_type_src}.pt"
        if not os.path.exists(outf_name):
            src_emb = make_custom_dir(src_custom, sent_type_src, api_key, org_key, custom_sentences_src)
            torch.save(src_emb, outf_name)
        else:
            src_emb = torch.load(outf_name)
    else:
        src_emb = hf_get_emb(d_desc2name[src])
    
    if dest=="make your own!":
        outf_name = f"tmp/template_emb_{dest_custom}_{sent_type_dest}.pt"
        if not os.path.exists(outf_name):
            dest_emb = make_custom_dir(dest_custom, sent_type_dest, api_key, org_key, custom_sentences_dest)
            torch.save(dest_emb, outf_name)
        else:
            dest_emb = torch.load(outf_name)
    else:
        dest_emb = hf_get_emb(d_desc2name[dest])
    text_dir = (dest_emb.cuda() - src_emb.cuda())*edit_mul

    if img_in_real is not None and img_in_synth is None:
        print("using real image")
        # resize the image so that the longer side is 512
        width, height = img_in_real.size
        if width > height: scale_factor = 512/width
        else: scale_factor = 512/height
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img_in_real = img_in_real.resize(new_size, Image.Resampling.LANCZOS)
        hash = hashlib.sha256(img_in_real.tobytes()).hexdigest()
        inv_fname = f"tmp/{hash}_ddim_{num_ddim}_inv.pt"
        caption_fname = f"tmp/{hash}_caption.txt"

        # make the caption if it hasn't been made before
        if not os.path.exists(caption_fname):
            # BLIP
            model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device("cuda"))
            _image = vis_processors["eval"](img_in_real).unsqueeze(0).cuda()
            prompt_str = model_blip.generate({"image": _image})[0]
            del model_blip
            torch.cuda.empty_cache()
            with open(caption_fname, "w") as f:
                f.write(prompt_str)
        else:
            prompt_str = open(caption_fname, "r").read().strip()
        print(f"CAPTION: {prompt_str}")
        
        # do the inversion if it hasn't been done before
        if not os.path.exists(inv_fname):
            # inversion pipeline
            pipe_inv = DDIMInversion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
            pipe_inv.scheduler = DDIMInverseScheduler.from_config(pipe_inv.scheduler.config)
            x_inv, x_inv_image, x_dec_img = pipe_inv( prompt_str, 
                    guidance_scale=1, num_inversion_steps=num_ddim,
                    img=img_in_real, torch_dtype=torch.float32 )
            x_inv = x_inv.detach()
            torch.save(x_inv, inv_fname)
            del pipe_inv
            torch.cuda.empty_cache()
        else:
            x_inv = torch.load(inv_fname)

        # do the editing
        edit_pipe = EditingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
        edit_pipe.scheduler = DDIMScheduler.from_config(edit_pipe.scheduler.config)

        _, edit_pil = edit_pipe(prompt_str,
                num_inference_steps=num_ddim,
                x_in=x_inv,
                edit_dir=text_dir,
                guidance_amount=xa_guidance,
                guidance_scale=5.0,
                negative_prompt=prompt_str # use the unedited prompt for the negative prompt
        )
        del edit_pipe
        torch.cuda.empty_cache()
        return edit_pil[0]

    elif img_in_real is None and img_in_synth is not None:
        print("using synthetic image")
        x_inv = torch.load(fpath_z_gen)
        pipe = EditingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        rec_pil, edit_pil = pipe(gen_prompt,
            num_inference_steps=num_ddim,
            x_in=x_inv,
            edit_dir=text_dir,
            guidance_amount=xa_guidance,
            guidance_scale=5,
            negative_prompt="" # use the empty string for the negative prompt
        )
        del pipe
        torch.cuda.empty_cache()
        return edit_pil[0]
    else:
        raise ValueError(f"Invalid image type found: {img_in_real} {img_in_synth}")



def set_visible_true():
    return gr.update(visible=True)

def set_visible_false():
    return gr.update(visible=False)


CSS_main = """
    body {
    font-family: "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif; 
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 10px;
    padding-right: 10px;
    width: 800px;
    }

    h1 {
        font-size:32px;
        font-weight:300;
        text-align: center;
    }

    h2 {
        font-size:32px;
        font-weight:300;
        text-align: center;
    }

    #lbl_gallery_input{
        font-family: 'Helvetica', 'Arial', sans-serif;
        text-align: center;
        color: #fff;
        font-size: 28px;
        display: inline
    }


    #lbl_gallery_comparision{
        font-family: 'Helvetica', 'Arial', sans-serif;
        text-align: center;
        color: #fff;
        font-size: 28px;
    }

    .disclaimerbox {
        background-color: #eee;		
        border: 1px solid #eeeeee;
        border-radius: 10px ;
        -moz-border-radius: 10px ;
        -webkit-border-radius: 10px ;
        padding: 20px;
    }

    video.header-vid {
        height: 140px;
        border: 1px solid black;
        border-radius: 10px ;
        -moz-border-radius: 10px ;
        -webkit-border-radius: 10px ;
    }

    img.header-img {
        height: 140px;
        border: 1px solid black;
        border-radius: 10px ;
        -moz-border-radius: 10px ;
        -webkit-border-radius: 10px ;
    }

    img.rounded {
        border: 1px solid #eeeeee;
        border-radius: 10px ;
        -moz-border-radius: 10px ;
        -webkit-border-radius: 10px ;
    }

    a:link
    {
        color: #941120;
        text-decoration: none;
    }
    a:visited
    {
        color: #941120;
        text-decoration: none;
    }
    a:hover {
        color: #941120;
    }

    td.dl-link {
        height: 160px;
        text-align: center;
        font-size: 22px;
    }

    .layered-paper-big { /* modified from: http://css-tricks.com/snippets/css/layered-paper/ */
        box-shadow:
        0px 0px 1px 1px rgba(0,0,0,0.35), /* The top layer shadow */
        5px 5px 0 0px #fff, /* The second layer */
        5px 5px 1px 1px rgba(0,0,0,0.35), /* The second layer shadow */
        10px 10px 0 0px #fff, /* The third layer */
        10px 10px 1px 1px rgba(0,0,0,0.35), /* The third layer shadow */
        15px 15px 0 0px #fff, /* The fourth layer */
        15px 15px 1px 1px rgba(0,0,0,0.35), /* The fourth layer shadow */
        20px 20px 0 0px #fff, /* The fifth layer */
        20px 20px 1px 1px rgba(0,0,0,0.35), /* The fifth layer shadow */
        25px 25px 0 0px #fff, /* The fifth layer */
        25px 25px 1px 1px rgba(0,0,0,0.35); /* The fifth layer shadow */
        margin-left: 10px;
        margin-right: 45px;
    }

    .paper-big { /* modified from: http://css-tricks.com/snippets/css/layered-paper/ */
        box-shadow:
        0px 0px 1px 1px rgba(0,0,0,0.35); /* The top layer shadow */

        margin-left: 10px;
        margin-right: 45px;
    }


    .layered-paper { /* modified from: http://css-tricks.com/snippets/css/layered-paper/ */
        box-shadow:
        0px 0px 1px 1px rgba(0,0,0,0.35), /* The top layer shadow */
        5px 5px 0 0px #fff, /* The second layer */
        5px 5px 1px 1px rgba(0,0,0,0.35), /* The second layer shadow */
        10px 10px 0 0px #fff, /* The third layer */
        10px 10px 1px 1px rgba(0,0,0,0.35); /* The third layer shadow */
        margin-top: 5px;
        margin-left: 10px;
        margin-right: 30px;
        margin-bottom: 5px;
    }

    .vert-cent {
        position: relative;
        top: 50%;
        transform: translateY(-50%);
    }

    hr
    {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
    }

    .card {
        /* width: 130px;
        height: 195px;
        width: 1px;
        height: 1px; */
        position: relative;
        display: inline-block;
        /* margin: 50px; */
    }
    .card .img-top {
        display: none;
        position: absolute;
        top: 0;
        left: 0;
        z-index: 99;
    }
    .card:hover .img-top {
        display: inline;
    }
    details {
    user-select: none;
    }

    details>summary span.icon {
    width: 24px;
    height: 24px;
    transition: all 0.3s;
    margin-left: auto;
    }

    details[open] summary span.icon {
    transform: rotate(180deg);
    }

    summary {
    display: flex;
    cursor: pointer;
    }

    summary::-webkit-details-marker {
    display: none;
    }

    ul {
    display: table;
    margin: 0 auto;
    text-align: left;
    }

    .dark {
        padding: 1em 2em;
        background-color: #333;
        box-shadow: 3px 3px 3px #333;
        border: 1px #333;
    }
    .column {
        float: left;
        width: 20%;
        padding: 0.5%;
    }

    .galleryImg {
    transition: opacity 0.3s;
    -webkit-transition: opacity 0.3s;
    filter: grayscale(100%);
    /* filter: blur(2px); */
    -webkit-transition : -webkit-filter 250ms linear;
    /* opacity: 0.5; */
    cursor: pointer; 
    }



    .selected {	
    /* outline: 100px solid var(--hover-background) !important; */
    /* outline-offset: -100px; */
    filter: grayscale(0%);
    -webkit-transition : -webkit-filter 250ms linear;
    /*opacity: 1.0 !important; */
    }
    
    .galleryImg:hover {
    filter: grayscale(0%);
    -webkit-transition : -webkit-filter 250ms linear;

    }

    .row {
    margin-bottom: 1em;
    padding: 0px 1em;
    }
    /* Clear floats after the columns */
    .row:after {
    content: "";
    display: table;
    clear: both;
    }
    
    /* The expanding image container */
    #gallery {
    position: relative;
    /*display: none;*/
    }

    #section_comparison{
        position: relative;
        width: 100%;
        height: max-content;
    }

    /* SLIDER
    -------------------------------------------------- */

    .slider-container {
        position: relative;
        height: 384px;
        width: 512px;
        cursor: grab;
        overflow: hidden;
        margin: auto;
    }
    .slider-after {
        display: block;
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    .slider-before {
        display: block;
        position: absolute;
        top: 0;
        /* right: 0; */
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 15;
        overflow: hidden;
    }
    .slider-before-inset {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
    }
    .slider-after img,
    .slider-before img {
        object-fit: cover;
        position: absolute;
        width: 100%;
        height: 100%;
        object-position: 50% 50%;
        top: 0;
        bottom: 0;
        left: 0;
        -webkit-user-select: none;
        -khtml-user-select: none;
        -moz-user-select: none;
        -o-user-select: none;
        user-select: none;
    }

    #lbl_inset_left{
        text-align: center;
        position: absolute;
        top: 384px;
        width: 150px;
        left: calc(50% - 256px);
        z-index: 11;
        font-size: 16px;
        color: #fff;
        margin: 10px;
    }
    .inset-before {
        position: absolute;
        width: 150px;
        height: 150px;
        box-shadow: 3px 3px 3px #333;
        border: 1px #333;
        border-style: solid;
        z-index: 16;
        top: 410px;
        left: calc(50% - 256px);
        margin: 10px;
        font-size: 1em;
        background-repeat: no-repeat;
        pointer-events: none;
    }

    #lbl_inset_right{
        text-align: center;
        position: absolute;
        top: 384px;
        width: 150px;
        right: calc(50% - 256px);
        z-index: 11;
        font-size: 16px;
        color: #fff;
        margin: 10px;
    }
    .inset-after {
        position: absolute;
        width: 150px;
        height: 150px;
        box-shadow: 3px 3px 3px #333;
        border: 1px #333;
        border-style: solid;
        z-index: 16;
        top: 410px;
        right: calc(50% - 256px);
        margin: 10px;
        font-size: 1em;
        background-repeat: no-repeat;
        pointer-events: none;
    }

    #lbl_inset_input{
        text-align: center;
        position: absolute;
        top: 384px;
        width: 150px;
        left: calc(50% - 256px + 150px + 20px);
        z-index: 11;
        font-size: 16px;
        color: #fff;
        margin: 10px;
    }
    .inset-target {
        position: absolute;
        width: 150px;
        height: 150px;
        box-shadow: 3px 3px 3px #333;
        border: 1px #333;
        border-style: solid;
        z-index: 16;
        top: 410px;
        right: calc(50% - 256px + 150px + 20px);
        margin: 10px;
        font-size: 1em;
        background-repeat: no-repeat;
        pointer-events: none;
    }

    .slider-beforePosition {
        background: #121212;
        color: #fff;
        left: 0;
        pointer-events: none;
        border-radius: 0.2rem;
        padding: 2px 10px;
    }
    .slider-afterPosition {
        background: #121212;
        color: #fff;
        right: 0;
        pointer-events: none;
        border-radius: 0.2rem;
        padding: 2px 10px;
    }
    .beforeLabel {
        position: absolute;
        top: 0;
        margin: 1rem;
        font-size: 1em;
        -webkit-user-select: none;
        -khtml-user-select: none;
        -moz-user-select: none;
        -o-user-select: none;
        user-select: none;
    }
    .afterLabel {
        position: absolute;
        top: 0;
        margin: 1rem;
        font-size: 1em;
        -webkit-user-select: none;
        -khtml-user-select: none;
        -moz-user-select: none;
        -o-user-select: none;
        user-select: none;
    }

    .slider-handle {
        height: 41px;
        width: 41px;
        position: absolute;
        left: 50%;
        top: 50%;
        margin-left: -20px;
        margin-top: -21px;
        border: 2px solid #fff;
        border-radius: 1000px;
        z-index: 20;
        pointer-events: none;
        box-shadow: 0 0 10px rgb(12, 12, 12);
    }
    .handle-left-arrow,
    .handle-right-arrow {
        width: 0;
        height: 0;
        border: 6px inset transparent;
        position: absolute;
        top: 50%;
        margin-top: -6px;
    }
    .handle-left-arrow {
        border-right: 6px solid #fff;
        left: 50%;
        margin-left: -17px;
    }
    .handle-right-arrow {
        border-left: 6px solid #fff;
        right: 50%;
        margin-right: -17px;
    }
    .slider-handle::before {
        bottom: 50%;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgb(12, 12, 12);
    }
    .slider-handle::after {
        top: 50%;
        margin-top: 20.5px;
        box-shadow: 0 0 5px rgb(12, 12, 12);
    }
    .slider-handle::before,
    .slider-handle::after {
        content: " ";
        display: block;
        width: 2px;
        background: #fff;
        height: 9999px;
        position: absolute;
        left: 50%;
        margin-left: -1.5px;
    }  


    /* 
    -------------------------------------------------
    The editing results shown below inversion results
    -------------------------------------------------
    */
    .edit_labels{
        font-weight:500;
        font-size: 24px;
        color: #fff;
        height: 20px;
        margin-left: 20px;
        position: relative; 
        top: 20px;
    }

    .open > a:hover {
        color: #555;
        background-color: red;
    }

    #directions { padding-top:30; padding-bottom:0; margin-bottom: 0px; height: 20px; }
    #custom_task { padding-top:0; padding-bottom:0; margin-bottom: 0px; height: 20px; }
    #slider_ddim {accent-color: #941120;}
    #slider_ddim::-webkit-slider-thumb {background-color: #941120;}
    #slider_xa {accent-color: #941120;}
    #slider_xa::-webkit-slider-thumb {background-color: #941120;}
    #slider_edit_mul {accent-color: #941120;}
    #slider_edit_mul::-webkit-slider-thumb {background-color: #941120;}

    #input_image [data-testid="image"]{
        height: unset;
    }
    #input_image_synth [data-testid="image"]{
        height: unset;
    }
"""


HTML_header = f"""
    <body>
    <center>
    <span style="font-size:36px">Zero-shot Image-to-Image Translation</span>
    <table align=center>
        <tr>
            <td align=center>
                <center>
                    <span style="font-size:24px; margin-left: 0px;"><a href='https://pix2pixzero.github.io/'>[Website]</a></span>
                    <span style="font-size:24px; margin-left: 20px;"><a href='https://github.com/pix2pixzero/pix2pix-zero'>[Code]</a></span>
                </center>
            </td>
        </tr>
    </table>
    </center>

    <center>
    <div align=center>
        <p align=left>
        This is a demo for <span style="font-weight: bold;">pix2pix-zero</span>, a diffusion-based image-to-image approach that allows users to 
        specify the edit direction on-the-fly (e.g., cat to dog). Our method can directly use pre-trained text-to-image diffusion models, such as Stable Diffusion, for editing real and synthetic images while preserving the input image's structure. Our method is training-free and prompt-free, as it requires neither manual text prompting for each input image nor costly fine-tuning for each task.
        <br>
        <span style="font-weight: 800;">TL;DR:</span> <span style=" color: #941120;"> no finetuning</span>  required; <span style=" color: #941120;"> no text input</span> needed; input <span style=" color: #941120;"> structure preserved</span>. 
        </p>
    </div>
    </center>


    <hr>
    </body>
"""


HTML_input_header = f"""
    <p style="font-size:150%; padding: 0px">
    <span font-weight: 800; style=" color: #941120;"> Step 1: </span> select a real input image.
    </p>
"""


HTML_middle_header = f"""
    <p style="font-size:150%;">
    <span font-weight: 800; style=" color: #941120;"> Step 2: </span> select the editing options.
    </p>
"""


HTML_output_header = f"""
    <p style="font-size:150%;">
    <span font-weight: 800; style=" color: #941120;"> Step 3: </span> translated image!
    </p>
"""

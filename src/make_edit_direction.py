import os, pdb

import argparse
import numpy as np
import torch
import requests
from PIL import Image

from diffusers import DDIMScheduler
from utils.edit_pipeline import EditingPipeline


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

## convert sentences to sentence embeddings
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device=device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_source_sentences', required=True)
    parser.add_argument('--file_target_sentences', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    args = parser.parse_args()

    # load the model
    if torch.cuda.is_available():
        pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    else:
        pipe = EditingPipeline.from_pretrained(args.model_path).to(device)

    bname_src = os.path.basename(args.file_source_sentences).strip(".txt")
    outf_src = os.path.join(args.output_folder, bname_src+".pt")
    if os.path.exists(outf_src):
        print(f"Skipping source file {outf_src} as it already exists")
    else:
        with open(args.file_source_sentences, "r") as f:
            l_sents = [x.strip() for x in f.readlines()]
        mean_emb = load_sentence_embeddings(l_sents, pipe.tokenizer, pipe.text_encoder, device=device)
        print(mean_emb.shape)
        torch.save(mean_emb, outf_src)

    bname_tgt = os.path.basename(args.file_target_sentences).strip(".txt")
    outf_tgt = os.path.join(args.output_folder, bname_tgt+".pt")
    if os.path.exists(outf_tgt):
        print(f"Skipping target file {outf_tgt} as it already exists")
    else:
        with open(args.file_target_sentences, "r") as f:
            l_sents = [x.strip() for x in f.readlines()]
        mean_emb = load_sentence_embeddings(l_sents, pipe.tokenizer, pipe.text_encoder, device=device)
        print(mean_emb.shape)
        torch.save(mean_emb, outf_tgt)

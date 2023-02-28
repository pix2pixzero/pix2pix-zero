import os
import pdb
from PIL import Image
import gradio as gr

from src.utils.gradio_utils import *
from src.utils.huggingface_utils import *
# from utils.generate_synthetic import *


if __name__=="__main__":
    # populate the list of editing directions
    d_name2desc = hf_get_all_directions_names()
    d_name2desc["make your own!"] = "make your own!"

    with gr.Blocks(css=CSS_main) as demo:
        # Make the header of the demo website
        gr.HTML(HTML_header)

        with gr.Row():
            # col A: the input image or synthetic image prompt
            with gr.Column(scale=2) as gc_left:
                gr.HTML(" <center> <p style='font-size:150%;'> input </p> </center>")
                img_in_real = gr.Image(type="pil", label="Start by uploading an image", elem_id="input_image")
                img_in_synth = gr.Image(type="pil", label="Synthesized image", elem_id="input_image_synth", visible=False)
                gr.Examples( examples="assets/test_images/cats", inputs=[img_in_real])
                prompt = gr.Textbox(value="a high resolution painting of a cat in the style of van gogh", label="Or use a synthetic image. Prompt:", interactive=True)
                with gr.Row():
                    seed = gr.Number(value=42, label="random seed:", interactive=True)
                    negative_guidance = gr.Number(value=5, label="negative guidance:", interactive=True)
                btn_generate = gr.Button("Generate", label="")
                fpath_z_gen = gr.Textbox(value="placeholder", visible=False)

            # col B: the output image
            with gr.Column(scale=2) as gc_left:
                gr.HTML(" <center> <p style='font-size:150%;'> output </p> </center>")
                img_out = gr.Image(type="pil", label="Output Image", visible=True)
                with gr.Row():
                    with gr.Column():
                        src = gr.Dropdown(list(d_name2desc.values()), label="source", interactive=True, value="cat")
                        src_custom = gr.Textbox(placeholder="enter new task here!", interactive=True, visible=False, label="custom source direction:")
                        rad_src = gr.Radio(["GPT3", "flan-t5-xl (free)!", "BLOOMZ-7B (free)!", "fixed-template", "custom sentences"], label="Sentence type:", value="GPT3", interactive=True, visible=False)
                        custom_sentences_src = gr.Textbox(placeholder="paste list of sentences here", interactive=True, visible=False, label="custom sentences:", lines=5, max_lines=20)
                    
                    with gr.Column():
                        dest = gr.Dropdown(list(d_name2desc.values()), label="target", interactive=True, value="dog")
                        dest_custom = gr.Textbox(placeholder="enter new task here!", interactive=True, visible=False, label="custom target direction:")
                        rad_dest = gr.Radio(["GPT3", "flan-t5-xl (free)!", "BLOOMZ-7B (free)!", "fixed-template", "custom sentences"], label="Sentence type:", value="GPT3", interactive=True, visible=False)
                        custom_sentences_dest = gr.Textbox(placeholder="paste list of sentences here", interactive=True, visible=False, label="custom sentences:", lines=5, max_lines=20)
                    
                    with gr.Row():
                        api_key = gr.Textbox(placeholder="enter you OpenAI API key here", interactive=True, visible=False, label="OpenAI API key:", type="password")
                        org_key = gr.Textbox(placeholder="enter you OpenAI organization key here", interactive=True, visible=False, label="OpenAI Organization:", type="password")
                with gr.Row():
                    btn_edit = gr.Button("Run", label="")
                    # btn_clear = gr.Button("Clear")

                with gr.Accordion("Change editing settings?", open=True):
                    num_ddim = gr.Slider(0, 200, 100, label="Number of DDIM steps", interactive=True, elem_id="slider_ddim", step=10)
                    xa_guidance = gr.Slider(0, 0.25, 0.1, label="Cross Attention guidance", interactive=True, elem_id="slider_xa", step=0.01)
                    edit_mul = gr.Slider(0, 2, 1.0, label="Edit multiplier", interactive=True, elem_id="slider_edit_mul", step=0.05)

                with gr.Accordion("Generating your own directions", open=False):
                    gr.Textbox("We provide 5 different ways of computing new custom directions:", show_label=False)
                    gr.Textbox("We use GPT3 to generate a list of sentences that describe the desired edit. For this options, the users need to make an OpenAI account and enter the API and organizations keys. This option typically results is the best directions and costs roughly $0.14 for one concept.", label="1. GPT3", show_label=True)
                    gr.Textbox("Alternatively flan-t5-xl model can also be used to to generate a list of sentences that describe the desired edit. This option is free and does not require creating any new accounts.", label="2. flan-t5-xl (free)", show_label=True)
                    gr.Textbox("Similarly BLOOMZ-7B model can also be used to to generate the sentences for free.", label="3. BLOOMZ-7B (free)", show_label=True)
                    gr.Textbox("Next, we provide a fixed template based sentence generation. This option does not require any language model and is therefore free and much faster. However the edit directions with this method are often entangled.", label="4. Fixed template", show_label=True)
                    gr.Textbox("Finally, the user can also generate their own sentences.", label="5. Custom sentences", show_label=True)

                with gr.Accordion("Tips for getting better results", open=True):
                    gr.Textbox("The 'Cross Attention guidance' controls the amount of structure guidance to be applied when performing the edit. If the output edited image does not retain the structure from the input, increasing the value will typically address the issue. We recommend changing the value in increments of 0.05.", label="1. Controlling the image structure", show_label=True)
                    gr.Textbox("If the output image quality is low or has some artifacts, using more steps would be helpful. This can be controlled with the 'Number of DDIM steps' slider.", label="2. Improving Image Quality", show_label=True)
                    gr.Textbox("There can be two reasons why the output image does not have the desired edit applied. Either the cross attention guidance is too strong, or the edit is insufficient. These can be addressed by reducing the 'Cross Attention guidance' slider or increasing the 'Edit multiplier' respectively.", label="3. Amount of edit applied", show_label=True)

        btn_generate.click(launch_generate_sample, [prompt, seed, negative_guidance, num_ddim], [img_in_synth, fpath_z_gen])
        def fn_set_none():
            return gr.update(value=None)
        btn_generate.click(fn_set_none, [], img_in_real)
        btn_generate.click(set_visible_true, [], img_in_synth)
        btn_generate.click(set_visible_false, [], img_in_real)


        def fn_clear_all():
            return gr.update(value=None), gr.update(value=None), gr.update(value=None)
        
        img_in_real.clear(fn_clear_all, [], [img_out, img_in_real, img_in_synth])
        img_in_real.clear(set_visible_true, [], img_in_synth)
        img_in_real.clear(set_visible_false, [], img_in_real)

        img_in_synth.clear(fn_clear_all, [], [img_out, img_in_real, img_in_synth])
        img_in_synth.clear(set_visible_true, [], img_in_real)
        img_in_synth.clear(set_visible_false, [], img_in_synth)

        img_out.clear(fn_clear_all, [], [img_out, img_in_real, img_in_synth])


        
        # handling custom directions
        def on_custom_seleceted(src):
            if src=="make your own!": return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            else: return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        src.change(on_custom_seleceted, [src], [src_custom, rad_src, api_key, org_key])
        dest.change(on_custom_seleceted, [dest], [dest_custom, rad_dest, api_key, org_key])


        def fn_sentence_type_change(rad):
            if rad=="GPT3":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
            elif rad=="custom sentences":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            else:
                print("using template sentence or flan-t5-xl or bloomz-7b")
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        rad_dest.change(fn_sentence_type_change, [rad_dest], [api_key, org_key, custom_sentences_dest])
        rad_src.change(fn_sentence_type_change, [rad_src], [api_key, org_key, custom_sentences_src])

        btn_edit.click(launch_main,
                        [
                            img_in_real, img_in_synth,
                            src, src_custom, dest,
                            dest_custom, num_ddim,
                            xa_guidance, edit_mul,
                            fpath_z_gen, prompt,
                            rad_src, rad_dest,
                            api_key, org_key,
                            custom_sentences_src, custom_sentences_dest
                        ],
                [img_out]
        )
        gr.HTML("<hr>")

    gr.close_all()
    demo.queue(concurrency_count=1)
    demo.launch(server_port=8088, server_name="0.0.0.0", debug=True)

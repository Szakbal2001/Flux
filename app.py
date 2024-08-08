import torch
from diffusers import FluxPipeline
import gradio as gr
import random
import numpy as np
import os
import spaces


if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


# login hf token
HF_TOKEN = os.getenv("HF_TOKEN")


MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"

# Initialize the pipeline and download the model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to(device)

# Enable memory optimizations
pipe.enable_attention_slicing()


# Define the image generation function
@spaces.GPU(duration=180)
def generate_image(prompt, num_inference_steps, height, width, guidance_scale, seed, num_images_per_prompt, progress=gr.Progress(track_tqdm=True)):
    if seed == 0:
        seed = random.randint(1, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)
    
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt
        ).images
    
    return output



# Create the Gradio interface

examples = [
    ["A cat holding a sign that says hello world"],
    ["a tiny astronaut hatching from an egg on the moon"],
    ["An astrounat on mars in a futuristic cyborg suit."],
]

css = '''
.gradio-container{max-width: 1000px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(
            """
            <h1 style='text-align: center'>
            FLUX.1-dev
            </h1>
            """
        )
            gr.HTML(
                """
               Made by <a href='https://linktr.ee/Nick088' target='_blank'>Nick088</a>
               <br> <a href="https://discord.gg/osai"> <img src="https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge" alt="Discord"> </a>
                """
        )
    with gr.Group():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", info="Describe the image you want", placeholder="A cat...")
            run_button = gr.Button("Run")
        result = gr.Gallery(label="Generated AI Images", elem_id="gallery")
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            num_inference_steps = gr.Slider(label="Number of Inference Steps", info="The number of denoising steps of the image. More denoising steps usually lead to a higher quality image at the cost of slower inference", minimum=1, maximum=50, value=25, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", info="Controls how much the image generation process follows the text prompt. Higher values make the image stick more closely to the input text.", minimum=0.0, maximum=7.0, value=3.5, step=0.1)
        with gr.Row():
            width = gr.Slider(label="Width", info="Width of the Image", minimum=256, maximum=1024, step=32, value=1024)
            height = gr.Slider(label="Height", info="Height of the Image", minimum=256, maximum=1024, step=32, value=1024)
        with gr.Row():
            seed = gr.Slider(value=42, minimum=0, maximum=MAX_SEED, step=1, label="Seed", info="A starting point to initiate the generation process, put 0 for a random one")
            num_images_per_prompt = gr.Slider(label="Images Per Prompt", info="Number of Images to generate with the settings",minimum=1, maximum=4, step=1, value=2)

    gr.Examples(
        examples=examples,
        fn=generate_image,
        inputs=[prompt, num_inference_steps, height, width, guidance_scale, seed, num_images_per_prompt],
        outputs=[result],
        cache_examples=CACHE_EXAMPLES
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate_image,
        inputs=[prompt, num_inference_steps, height, width, guidance_scale, seed, num_images_per_prompt],
        outputs=[result],
    )

demo.queue().launch(share=True)

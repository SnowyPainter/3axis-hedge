from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("nicky007/stable-diffusion-logo-fine-tuned")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
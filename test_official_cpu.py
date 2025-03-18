import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import ControlNetModel

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)

prompt = "a photo of an astronaut riding a horse on mars"
print(pipe.components)

pipe.enable_sequential_cpu_offload()
# # pipe.to("cuda")
# image = pipe(prompt, num_inference_steps=40).images[0]


from diffusers import StableDiffusionXLControlNetPipeline
from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
)
from PIL import Image

controlnet_path = f"./checkpoints/ControlNetModel"
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=torch.float16
)

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "wangqixun/YamerMIX_v8",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    controlnet=[controlnet_identitynet],
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.enable_model_cpu_offload()
image = pipe(prompt, image=[Image.open("Untitled.png")], num_inference_steps=40).images[
    0
]
images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    image=[Image.open("Untitled.png")],
    control_mask=control_mask,
    controlnet_conditioning_scale=control_scales,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    generator=generator,
)

print(type(image))

image.save("output.jpg")

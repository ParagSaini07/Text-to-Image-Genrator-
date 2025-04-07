from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the model to GPU if available (faster)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Enter your prompt here
prompt = "a cute cat wearing a hat, cartoon style"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")
print("Image saved as 'generated_image.png'!")
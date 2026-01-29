import base64
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Use your exact model loading code
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

def handler(event, context):
    """
    Serverless handler that accepts JSON {"prompt": "..."} and returns base64 PNG image.
    """
    prompt = event.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}

    print(f"Generating image for prompt: {prompt} ... (this may take a while)")
    image = pipe(prompt).images[0]

    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image_base64": img_str}

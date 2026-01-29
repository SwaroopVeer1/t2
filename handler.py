# handler.py
import os
import io
import base64
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Create image folder
os.makedirs("img", exist_ok=True)

# Load model once when the container starts
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
print("Model loaded.")

def handler(event, context):
    """
    RunPod serverless handler.
    Expects JSON: {"prompt": "Your text prompt here"}
    Returns base64 PNG image.
    """
    try:
        # Extract prompt
        prompt = event.get("prompt", "")
        if not prompt:
            return {"statusCode": 400, "body": "Error: 'prompt' is required."}

        # Generate image
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Save locally (optional)
        image_path = os.path.join("img", "generated.png")
        image.save(image_path)

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "statusCode": 200,
            "body": {"image_base64": img_str}
        }

    except Exception as e:
        return {"statusCode": 500, "body": str(e)}

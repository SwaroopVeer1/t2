import os
import base64
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
import runpod

print("Loading Stable Diffusion...")

# Load the model at startup
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded.")

def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 20) -> str:
    """Generate an image and return it as Base64 string."""
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps
        )
    img = result.images[0]
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(event):
    """
    Expected JSON input:
    {
        "input": {
            "prompt": "A cute cartoon robot",
            "width": 512,
            "height": 512,
            "steps": 20
        }
    }
    """
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "A serene landscape at sunrise")
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    steps = input_data.get("steps", 20)

    try:
        image_b64 = generate_image(prompt, width, height, steps)
        return {
            "output": {
                "image": image_b64,
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

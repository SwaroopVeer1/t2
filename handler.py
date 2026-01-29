import base64
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
import runpod

# ------------------------------
# Load the Stable Diffusion model at runtime (GPU required)
# ------------------------------
print("Loading Stable Diffusion model...")
model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Model loaded on {device}!")

# ------------------------------
# Helper function to generate Base64 image
# ------------------------------
def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 20) -> str:
    try:
        with torch.autocast("cuda"):
            image = pipe(prompt, height=height, width=width, num_inference_steps=steps).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return image_base64
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e

# ------------------------------
# Serverless handler
# ------------------------------
def handler(event):
    input_data = event.get("input", {})

    prompt = input_data.get("prompt", "A beautiful landscape")
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    steps = input_data.get("steps", 20)

    try:
        image_base64 = generate_image(prompt, width, height, steps)
        return {
            "output": {
                "image": image_base64,
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps
            }
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# Start RunPod serverless
# ------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

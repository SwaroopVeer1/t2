# Use NVIDIA PyTorch container for GPU support
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install diffusers transformers accelerate runpod

# Copy handler
COPY handler.py /app/handler.py
WORKDIR /app

# Run the handler
CMD ["python", "handler.py"]

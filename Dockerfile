# Use PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system tools
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install diffusers transformers accelerate runpod

# Copy the handler
COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "handler.py"]

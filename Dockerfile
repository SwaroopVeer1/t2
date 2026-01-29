# Use CUDA runtime for GPU support
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and git
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your handler and any other files
COPY . .

# Set handler for Runpod serverless
ENV HANDLER_MODULE=handler
ENV HANDLER_FUNCTION=handler

# Command to run serverless
CMD ["python3", "-m", "runpod.serverless"]

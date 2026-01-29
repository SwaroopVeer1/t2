FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python, git
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
# Also install Flask for HTML interface
RUN pip3 install flask

# Copy code
COPY . .

# Expose port for web access
EXPOSE 8080

# Run Flask app
CMD ["python3", "app.py"]

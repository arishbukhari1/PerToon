# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p assets/raw_humanface assets/memes assets/cartoonized outputs/captions logs

# Download models if not present (optional - can be done at runtime)
# RUN python models/download_vision_models.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface
EXPOSE 8080

# Default command to run the main pipeline
CMD ["python", "-c", "print('PerToon Docker container is ready! Run notebooks in scripts/ directory to execute the pipeline.')"]

# Alternative commands for specific tasks:
# To run the full pipeline: 
# CMD ["jupyter", "nbconvert", "--execute", "--to", "notebook", "scripts/3b_final_image_caption_integration.ipynb"]

# To start jupyter server:
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root"] 
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project
RUN pip install -e .

# Download pre-trained checkpoint (placeholder)
# RUN python scripts/download_checkpoints.py

# Expose port for potential API
EXPOSE 8000

CMD ["bash"]

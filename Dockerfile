# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Clone dependencies from source
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    pip install ./GroundingDINO
RUN git clone https://github.com/facebookresearch/segment-anything.git && \
    pip install ./segment-anything

# Copy source code
COPY . .

# Set default command
CMD ["python", "main.py"]

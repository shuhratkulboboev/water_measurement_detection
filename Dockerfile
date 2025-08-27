# Use Python base image instead of Ubuntu for smaller size
FROM python:3.10-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" \
    VIRTUAL_ENV=/app/venv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    opencv-python-headless \
    paho-mqtt \
    tensorflow

# Copy application files
COPY model/DigitDetector_130epochs.h5 /app/models/
COPY scripts/water_meter_v4.py /app/
COPY images /app/images
COPY videos /app/videos

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; import paho.mqtt.client as mqtt; print('Health check passed')" || exit 1

# Set the entrypoint
ENTRYPOINT ["python", "water_meter_v4.py"]


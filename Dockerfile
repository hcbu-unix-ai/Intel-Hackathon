# Base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /openvino

# Copy requirements.txt to the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the Docker container
COPY app /openvino/app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CACHE_DIR='/openvino/app/cache'
ENV DOCUMENT_DIR='/openvino/app/docs'
ENV CHROMA_PATH='/openvino/app/docs_embedding'

# Expose the ports where FastAPI and Streamlit servers will run
EXPOSE 8044 8504

# Set the working directory to the app folder where all scripts are located
WORKDIR /openvino/app

# Run the run.py script to start both servers
CMD ["python", "run.py"]

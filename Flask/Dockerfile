FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        --fix-missing
# Install any python packages you need
WORKDIR /app
COPY . .

RUN python3 -m pip install accelerate peft bitsandbytes transformers trl Flask

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --upgrade numpy

# Set the working directory
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
# Set the entrypoint
EXPOSE 5000
CMD ["flask", "run"] 
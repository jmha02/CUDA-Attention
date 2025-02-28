# Base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# Set up working directory
WORKDIR /workspace

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir ninja

# Clone the simple-flash-attention repository
# RUN git clone https://github.com/YWHyuk/simple-flash-attention.git
COPY . /workspace/simple-flash-attention

# Set entry point
CMD ["/bin/bash"]

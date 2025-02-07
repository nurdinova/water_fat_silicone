# Use CUDA 11.4 base image
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# Add NVIDIA CUDA repository
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install NCCL for CUDA 11.4
RUN apt-get update && \
    apt-get install -y libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4 && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib64/libnccl.so

# Install base dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim && \
    rm -rf /var/lib/apt/lists/*

# Update Python tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install CUDA-specific Python libraries
RUN pip3 install --no-cache-dir cupy-cuda11x

# Pin numpy version for compatibility
RUN pip3 install numpy==1.22.3

# Install additional Python libraries
RUN pip3 install finufft \
                 h5py \
                 ipympl \
                 jupyterlab \
                 jupytext \
                 matplotlib \
                 pydicom \
                 sigpy \
                 scikit-image \
                 sparse \
                 opencv-python \
                 tensorflow==2.11.0 \
                 tensorflow_io \
                 wget \
                 SimpleITK \
                 nibabel \
                 pandas \
                 Cython \
                 numba \
                 cufinufft \
                 ipywidgets==7.7.1

RUN pip install pymaxflow --no-build-isolation

# Install PyTorch with CUDA 11.4 support
RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu114
RUN pip install torchmetrics

# Clone and patch cufinufft (to fix Python interface issue)
RUN git clone https://github.com/flatironinstitute/cufinufft.git && \
    cp cufinufft/python/cufinufft/*.py /usr/local/lib/python3.8/dist-packages/cufinufft/ && \
    rm -rf cufinufft

# Add a non-root user
RUN adduser --disabled-password --gecos '' sdkuser && \
    mkdir /home/sdkuser/workspace

# Set up OpenSSL (for specific environments)
RUN wget https://www.openssl.org/source/old/1.0.1/openssl-1.0.1e.tar.gz -O - | tar xvz -C . && \
    cd openssl-1.0.1e && \
    ./config shared && \
    make && \
    make install_sw && \
    cd .. && \
    rm -rf openssl-1.0.1e && \
    ldconfig && \
    ln -s /usr/local/ssl/lib/libssl.so.1.0.0 /usr/local/ssl/lib/libssl.so.10 && \
    ln -s /usr/local/ssl/lib/libcrypto.so.1.0.0 /usr/local/ssl/lib/libcrypto.so.10 && \
    cp /usr/local/ssl/lib/libssl.so.10 /lib/x86_64-linux-gnu/libssl.so.10 && \
    cp /usr/local/ssl/lib/libcrypto.so.10 /lib/x86_64-linux-gnu/libcrypto.so.10

# Set environment variables
ENV PATH="$PATH:/home/sdkuser/workspace/bin/"

# Switch to non-root user
USER sdkuser

# Set working directory
WORKDIR /home/sdkuser/workspace

# Default command
CMD /bin/bash

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    gdb \
    time \
    vim \
    python3 \
    python3-pip \
    git \
    nvcc \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=4
WORKDIR /workspace

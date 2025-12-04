FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update & install common Linux tools
RUN apt update && apt upgrade -y && apt install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    vim \
    nano \
    zip \
    unzip \
    tar \
    sudo \
    software-properties-common

# Install Python
RUN apt install -y python3 python3-pip python3-venv

# Install Node.js (LTS 18 or 20)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt install -y nodejs

# Install Java (OpenJDK 17)
RUN apt install -y openjdk-17-jdk

# Create workspace
WORKDIR /workspace

# Default command: open a Linux shell
CMD ["/bin/bash"]
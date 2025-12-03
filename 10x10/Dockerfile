# Use official Ubuntu as base image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY alphazero.py /app/
COPY alphazero_utilize.py /app/
COPY game.py /app/
COPY config.yaml /app/
COPY temp /app/temp

# Create virtual environment and activate
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command (adjust as needed)
CMD ["python", "alphazero_utilize.py", "--train", "--wandb"]

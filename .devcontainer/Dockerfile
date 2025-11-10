FROM mcr.microsoft.com/devcontainers/python:3.11


# Set env variables
ENV PYTHONIOENCODING=UTF-8


# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential vim \
    && rm -rf /var/lib/apt/lists/*

#Create a workspace
WORKDIR /workspace

# Copy requirements file
COPY ../requirements.txt /workspace/requirements.txt
COPY ../src /workspace/src


# Install Python dependencies
RUN pip install --upgrade pip && pip install --upgrade -r /workspace/requirements.txt


# Set the default command to bash    

CMD ["python", "/workspace/src/main.py"]

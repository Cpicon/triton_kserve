# Use the official Python 3.11 image as the base
FROM python:3.11-slim as downloader

# Define build-time arguments
ARG MODEL_NAME=vit_base_patch16_384
ARG VERSION=1
ARG BACKEND=pytorch

# Set the working directory inside the container
WORKDIR /app

# Copy the local imageclassifier package into the container
COPY imageclassifier /app/imageclassifier

COPY pyproject.toml /app/pyproject.toml

# Copy the config.json file into the container
COPY deployment/dev/triton_server/config.json /app/config.json

# Install the imageclassifier package
RUN pip install .

# Run the create-repository command
RUN python /app/imageclassifier/clients.py create-repository $MODEL_NAME $VERSION $BACKEND /app/config.json

# Check if the directory exists
RUN test -d /app/model_repository/$MODEL_NAME/$VERSION || (echo "Directory does not exist" && exit 1)

# Run the download-model command
RUN python /app/imageclassifier/clients.py download-model $MODEL_NAME

# Second Stage: Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:24.10-pyt-python-py3 as triton_server

# Set the working directory inside the container
WORKDIR /models

# Copy the model repository from the first stage
COPY --from=downloader /app/model_repository /models

# Expose the Triton server ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Set the entrypoint to run the Triton server
ENTRYPOINT ["tritonserver", "--model-repository=/models"]
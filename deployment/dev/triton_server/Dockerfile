# Use the official Python 3.11 image as the base
FROM python:3.10-slim as downloader

# Define build-time arguments
ARG MODEL_NAME=vit_base_patch16_384
ARG VERSION=1
ARG BACKEND=pytorch

# Set the working directory inside the container
WORKDIR /app

# Copy the config.json file into the container
COPY deployment/dev/triton_server/image_classifier/config.json /app/image_classifier/config.json
COPY deployment/dev/triton_server/preprocessor/config.json /app/preprocessor/config.json

# Install the imageclassifier package
RUN pip install torch timm click

COPY imageclassifier/model_repository_cli.py /app/imageclassifier/model_repository_cli.py

# Run the create-repository command
RUN python /app/imageclassifier/model_repository_cli.py create-repository $MODEL_NAME $VERSION $BACKEND /app/image_classifier/config.json
RUN python /app/imageclassifier/model_repository_cli.py create-repository image_preprocessor 1 python /app/preprocessor/config.json

# Check if the directory exists
RUN test -d /app/model_repository/$MODEL_NAME/$VERSION || (echo "Directory does not exist" && exit 1)

COPY deployment/dev/triton_server/preprocessor/model.py /app/model_repository/image_preprocessor/1/model.py

# Run the download-model command
RUN python /app/imageclassifier/model_repository_cli.py download-model $MODEL_NAME

# Second Stage: Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:24.10-pyt-python-py3 as triton_server

# Copy the local imageclassifier package into the container
COPY imageclassifier /app/imageclassifier

COPY pyproject.toml /app/pyproject.toml

# Install the imageclassifier package
RUN cd /app/ && pip install .
# append package to the python path
ENV PYTHONPATH=/app:$PYTHONPATH

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
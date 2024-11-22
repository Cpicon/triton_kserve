# Use the official Python 3.11 image
FROM python:3.11-slim
ENV INFERENCE_SERVER=localhost:8000
# Set working directory inside the container
WORKDIR /app

ENV PYTHONPATH=/app:$PYTHONPATH
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY imageclassifier/ /app/imageclassifier
COPY pyproject.toml /app/pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose the default Streamlit port
EXPOSE 8501

# Set the entrypoint to Streamlit
CMD ["streamlit", "run", "/app/imageclassifier/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
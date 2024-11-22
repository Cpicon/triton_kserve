# Model Serving Challenge

This project involves using the **Triton Inference Server** to serve machine learning models, leveraging its PyTorch and Python backends. The primary objective is to familiarize  with Triton's basic usage while demonstrating the ability to work with model serving, preprocessing pipelines, and scalable deployment.

### Objectives
1. **Compile the Model:** Take the provided example model and compile it into the TorchScript format for optimized serving.
2. **Serve the Compiled Model:** Use Triton's PyTorch backend to serve the compiled model and ensure it can be accessed as a client.
3. **Create a Preprocessing Model:** Build a Python-based model for preprocessing. This should handle at least resizing, but you are encouraged to include any additional preprocessing steps that might enhance the pipeline.
4. **Create an Ensemble Model:** Combine the preprocessing model and the main model into an ensemble that runs both sequentially.

### Repository Structure
Explain the repository structure and the purpose of each file and directory.
```
deployment/
├── dev/                # Files for local development (e.g., Dockerfiles, configs)
├── stage/              # Configuration files for staging environments
└── prod/               # Deployment files for production environments

imageclassifier/
├── app/                # User interface 
├── models/             # Main model and preprocessing scripts
├── clients.py          # Client for interacting with Triton Server
└── __init__.py         # Package initialization

data/
└── test_data/          # Sample data for testing the model serving pipeline

tests/ # Unit tests for the project

model_repository/
└── vit_base_patch16_384/  # Model repository for Triton Server
    ├── 1/               # Versioned model directory
    └── config.pbtxt     # Triton configuration file
```


### Running the Project
you can run the project with docker or locally for development.

1. run the project with docker
    ```bash
    # download artifacts
    dvc pull
    #build docker image
    make build
    #run docker container
    make run
    # go to the ui http://localhost:8501, pick up an image from data/images folder, enjoy!
    #stop docker container
    make stop
    ```
2. local development
    ```bash
    #create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    #install dependencies
    pip install -r requirements.txt

    # create model repository and the config.pbtxt file
    python imageclassifier/model_repository_cli.py  create-repository vit_base_patch16_384 1 pytorch deployment/dev/triton_server/image_classifier/config.json
    python imageclassifier/model_repository_cli.py create-repository image_preprocessor 1 python deployment/dev/triton_server/preprocessor/config.json
    python imageclassifier/model_repository_cli.py create-repository ensemble_model 1 python deployment/dev/triton_server/ensemble_model/config.json
    #download model 
    python imageclassifier/model_repository_cli.py  download-model vit_base_patch16_384
    cp deployment/dev/triton_server/preprocessor/model.py model_repository/image_preprocessor/1/model.py

    #run triton server
    docker run -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/app nvcr.io/nvidia/tritonserver:24.10-pyt-python-py3
   
    #run the model server
    cd /app 
    pip install .
    tritonserver --model-repository=/app/model_repository
    ```

### Running Tests with Pytest
To ensure the integrity of the code and verify the functionality of the models and serving pipelines, you can run the tests using `pytest`. 

Run the following command from the root directory of the project:
```bash
pytest tests/
```

This will execute all unit tests within the `tests/` directory and provide a detailed report of test outcomes.
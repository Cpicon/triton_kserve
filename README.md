# Model Serving Challenge

This challenge involves using the **Triton Inference Server** to serve machine learning models, leveraging its PyTorch and Python backends. The primary objective is to familiarize yourself with Triton's basic usage while demonstrating your ability to work with model serving, preprocessing pipelines, and scalable deployment.

### Requirements
1. **Compile the Model:** Take the provided example model and compile it into the TorchScript format for optimized serving.
2. **Serve the Compiled Model:** Use Triton's PyTorch backend to serve the compiled model and ensure it can be accessed as a client.
3. **Create a Preprocessing Model:** Build a Python-based model for preprocessing. This should handle at least resizing, but you are encouraged to include any additional preprocessing steps that might enhance the pipeline.
4. **Create an Ensemble Model:** Combine the preprocessing model and the main model into an ensemble that runs both sequentially.
5. **Deploy on KServe:** Deploy the model on **KServe** to ensure scalability and reliability. Use **K3s** for local Kubernetes cluster setup during testing.

### Deliverables
1. **Instructions for Running Triton:** Provide clear, step-by-step instructions (or a bash script) to launch Triton and serve the models.
2. **Demonstration App:** Develop a simple application that interacts with the served models in Triton, either using Triton's client library or its HTTP interface.
3. **KServe Deployment Setup:** Include a Kubernetes deployment configuration for running the models in KServe. Ensure the deployment works locally with K3s and demonstrate how to interact with the model through KServe's API.

This challenge will test your understanding of model serving fundamentals, scalable deployment strategies, and your ability to set up a reliable serving pipeline using Triton and KServe.
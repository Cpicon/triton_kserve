import numpy as np
import tritonclient.http as httpclient
from PIL import ImageFile
from torchvision import transforms


def run_inference(
    image: ImageFile.ImageFile,
    model_name: str,
    classes: list[str],
    models: dict[str, dict[str, str]],
    server_url: str = "localhost:8000",
) -> tuple[str, str]:
    """
    Runs inference on a given image using the specified model on the Triton Inference Server.

    Args:
        image(ImageFile.ImageFile): input image file.
        model_name (str): Name of the model to use for inference.
        classes (List[str]): List of class names for prediction output.
        models (Dict[str, Dict[str, str]]): Configuration for models with input and output mappings.
        server_url (str, optional): URL of the Triton Inference Server. Defaults to "localhost:8000".

    Raises:
        ValueError: If the specified model is not found in the models configuration.
    return:
        Tuple[str, str]: Index and class name of the predicted output.
    """
    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' not found in the provided models configuration."
        )

    # Load and preprocess the image
    preprocess = transforms.Compose([transforms.ToTensor()])
    numpy_image = preprocess(image).numpy()

    # Configure model input and output
    config = models[model_name]
    inputs = httpclient.InferInput(
        config["input"], numpy_image.shape, datatype="FP32"
    )
    inputs.set_data_from_numpy(numpy_image, binary_data=True)

    # Perform inference
    with httpclient.InferenceServerClient(server_url) as client:
        response = client.infer(model_name, [inputs])
        output = response.as_numpy(config["output"])

    # Display results
    predicted_index = np.argmax(output[0])
    predicted_class = classes[predicted_index]

    return predicted_index, predicted_class

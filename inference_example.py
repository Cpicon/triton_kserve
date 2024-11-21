from PIL import Image

from imageclassifier import run_inference
import os
if __name__ == "__main__":
    CLASSES = [
        "house",
        "tree",
        "bunny",
        "turtle",
        "storm",
        "record-player",
        "ron-howard",
    ]

    MODELS = {
        "ensemble_model": {"input": "input_image", "output": "probabilities_output"},
        "image_preprocessor": {"input": "input_image", "output": "processed_image"},
        "image_classifier": {"input": "processed_image", "output": "probabilities_output"},
    }

    IMAGE_PATH = "01.jpg"
    SERVER_URL = os.environ.get("INFERENCE_SERVER", "localhost:8000")
    MODEL_NAME = "ensemble_model"
    image = Image.open(IMAGE_PATH)
    run_inference(
        image=image,
        model_name=MODEL_NAME,
        classes=CLASSES,
        models=MODELS,
        server_url=SERVER_URL,
    )


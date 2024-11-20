from typing import Any

import numpy as np


def execute_feature_extractor(
    feature_extractor: Any, image: np.ndarray
) -> np.ndarray:
    inputs = feature_extractor(images=image, return_tensors="pt")
    transformed_img = inputs.pixel_values.cpu().numpy()
    return transformed_img

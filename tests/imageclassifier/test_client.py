from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_inference_server_client():
    with patch(
        "imageclassifier.client.httpclient.InferenceServerClient",
        autospec=True,
    ) as mock_client:
        client_instance = mock_client.return_value
        response_mock = Mock()
        response_mock.as_numpy.return_value = np.array(
            [[0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0]]
        )
        client_instance.infer.return_value = response_mock
        yield mock_client


@pytest.fixture
def mock_transforms_to_tensor():
    with patch(
        "imageclassifier.client.transforms.Compose.numpy", autospec=True
    ) as mock_tensor:
        tensor_mock = Mock()
        tensor_mock.numpy.return_value = np.zeros(
            (3, 384, 384), dtype=np.float32
        )
        mock_tensor.return_value = tensor_mock
        yield mock_tensor


@pytest.fixture
def setup_inputs():
    classes = [
        "house",
        "tree",
        "bunny",
        "turtle",
        "storm",
        "record-player",
        "ron-howard",
    ]
    models = {
        "ensemble_model": {
            "input": "input_image",
            "output": "probabilities_output",
        },
    }
    return {
        "image": Mock(),  # Mock PIL image
        "model_name": "ensemble_model",
        "classes": classes,
        "models": models,
        "server_url": "localhost:8000",
    }

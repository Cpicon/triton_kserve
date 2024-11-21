from unittest.mock import Mock

import numpy as np
import pytest

from imageclassifier.features.image_preprocessor import execute_feature_extractor


@pytest.fixture
def mock_feature_extractor():
    """Fixture to create a mock feature extractor."""
    mock = Mock()
    # Simulate the return structure of the feature extractor
    mock.return_value = Mock()
    mock.return_value.pixel_values = Mock()
    mock.return_value.pixel_values.cpu.return_value = Mock()
    mock.return_value.pixel_values.cpu.return_value.numpy.return_value = (
        np.array([[1, 2, 3]])
    )
    return mock


def test_execute_feature_extractor(mock_feature_extractor):
    """Test the execute_feature_extractor function."""
    # Create a sample image input
    test_image = np.random.rand(3, 256, 256).astype("float32")

    # Call the function with the mock feature extractor
    result = execute_feature_extractor(mock_feature_extractor, test_image)

    # Assertions to validate the behavior
    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert result.shape == (
        1,
        3,
    ), "Result shape does not match expected shape."

    # Ensure that the feature extractor was called with the correct arguments
    mock_feature_extractor.assert_called_once_with(
        images=test_image, return_tensors="pt"
    )

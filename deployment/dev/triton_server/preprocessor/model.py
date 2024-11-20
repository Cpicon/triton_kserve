import io

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from transformers import ViTImageProcessor

from imageclassifier.features.image_preprocessor import execute_feature_extractor


class TritonPythonModel:

    def initialize(self, args):
        """Initialize the model."""
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-384"
        )

    def execute(self, requests):
        """`execute` is called once for every inference request. This function
        must be implemented by the model. The function receives a list of
        TritonPythonRequest objects when invoked. The function must return a
        list of TritonPythonResponse objects.
        Parameters
        ----------
        requests : list of TritonPythonRequest
          A list of TritonPythonRequest objects. Each object contains the
          request for inference.
        Returns
        -------
        responses : list of TritonPythonResponse
          A list of TritonPythonResponse objects.
        """
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "image_preprocessor_input"
            )
            img = in_0.as_numpy()
            transformed_img = execute_feature_extractor(self.feature_extractor, img)
            out_tensor_0 = pb_utils.Tensor(
                "image_preprocessor_output", transformed_img.astype("float32")
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        return responses
import os

import streamlit as st
from PIL import Image

from imageclassifier import run_inference

# Define constants
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
    "ensemble_model": {
        "input": "input_image",
        "output": "probabilities_output",
    },
}
DEFAULT_SERVER_URL = os.getenv("INFERENCE_SERVER", "localhost:8000")
MODEL_NAME = "ensemble_model"
UPLOAD_FOLDER = "uploaded_images"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit UI
st.title("Image Classification with Triton Inference Server")

# Sidebar for server URL configuration
server_url = st.sidebar.text_input(
    "Triton Inference Server URL", DEFAULT_SERVER_URL
)

# Option to upload an image
uploaded_image = st.file_uploader(
    "Upload an image for classification", type=["jpg", "jpeg", "png"]
)

# Option to select an image from a folder
folder_images = [
    f
    for f in os.listdir(UPLOAD_FOLDER)
    if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
]
selected_image = st.selectbox(
    "Or select an image from the folder:", [""] + folder_images
)

# Determine the image to use
image_to_use = None
if uploaded_image:
    image = Image.open(uploaded_image)
    image_to_use = image
    # Save uploaded image to the folder
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_image.name)
    image.save(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)
elif selected_image:
    image_path = os.path.join(UPLOAD_FOLDER, selected_image)
    image = Image.open(image_path)
    image_to_use = image
    st.image(image, caption="Selected Image", use_container_width=True)

# Perform inference when an image is selected
if st.button("Classify Image"):
    if image_to_use:
        try:
            with st.spinner("Running inference..."):
                # Call inference function
                predicted_index, predicted_class = run_inference(
                    image=image_to_use,
                    model_name=MODEL_NAME,
                    classes=CLASSES,
                    models=MODELS,
                    server_url=server_url,
                )
            # Beautify the prediction display
            st.markdown(
                f"<div style='text-align: center; font-size: 1.5rem; color: white;'>"
                f"<b>Predicted Class:</b> {predicted_class}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error during inference: {e}")
    else:
        st.warning(
            "Please upload or select an image before running inference."
        )

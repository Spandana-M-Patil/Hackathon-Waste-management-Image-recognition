import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer

# ---- Setup ----
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Image Classifier")

# ---- Class Names ----
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # change based on your model

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = TFSMLayer("trashClassifier", call_endpoint="serving_default")
    return model

model = load_model()

# ---- Image Upload ----
uploaded_file = st.file_uploader("Upload an image of trash", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    output = model(img_array)

    # If output is a dict (TFSMLayer), extract it
    if isinstance(output, dict):
        output = list(output.values())[0].numpy()
    else:
        output = output.numpy()

    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output)

    st.success(f"Predicted: **{predicted_class}** ({confidence * 100:.2f}% confidence)")

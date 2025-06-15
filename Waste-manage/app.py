import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Smart Garbage Segregation 🗑️")

@st.cache_resource
def load_model():
    model_path = "trashClassifier"  
    model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
    return model

model = load_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

disposal_guide = {
    "plastic": " *Plastic goes in the recycling bin.* Make sure to rinse bottles or containers before throwing them.",
    "glass": " *Glass should be recycled.* Clean it and avoid mixing with ceramics or broken mirrors.",
    "metal": " *Metal cans and containers are recyclable.* Rinse them and place in the recycling bin.",
    "paper": " *Paper is recyclable.* Keep it dry and away from food waste.",
    "cardboard": " *Cardboard should be flattened before recycling.* Avoid greasy ones like pizza boxes.",
    "trash": " *This is general waste.* It cannot be recycled. Put it in the regular trash bin.",
}

uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model(img_array)
    if isinstance(predictions, dict): 
        predictions = list(predictions.values())[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    st.success(f"Predicted: **{predicted_class.upper()}**")
    st.info(disposal_guide[predicted_class])

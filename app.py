import streamlit as st
from PIL import Image

st.title("Smart Garbage Segregation 🗑️")

# Upload an image
uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "png", "jpeg"])

# Show the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Image uploaded successfully!")

# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

st.title("🖤 MNIST Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a digit (0-9)")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((28, 28))  # resize to 28x28
    st.image(img, caption="Uploaded Image", width=150)

    # Preprocess
    img_array = np.array(img).reshape(1, 28*28).astype("float32") / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.subheader("Prediction")
    st.write(f"👉 The model predicts this is a **{pred_class}**")

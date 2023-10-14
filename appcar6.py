import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set the background color, text color, and font
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .stApp {
        max-width: 800px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your Streamlit app code
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cars_classifier.hdf5')
    return model

model = load_model()

st.title("Car Classifier")

file = st.file_uploader("Choose a car photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']
    result = class_names[np.argmax(prediction)]
    st.subheader("Prediction:")
    st.success(result)

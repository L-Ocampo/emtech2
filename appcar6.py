import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cars_classifier.hdf5')
    return model

model = load_model()

st.write("""
# Car Classifier
""")

file = st.file_uploader("Choose a car photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)  # Use Image.ANTIALIAS directly
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
    string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(string)

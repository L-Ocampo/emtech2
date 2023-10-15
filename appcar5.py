import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cars_classifier.hdf5')
    return model

# Page title and description
st.title("Car Classifier")
st.write("Upload a car photo, and I'll classify it into one of the following :")

# Sidebar with explanation and credit
st.sidebar.header("About")
st.sidebar.markdown("This app uses a deep learning model to classify car images.")
st.sidebar.markdown("Model credit: [Link to Model Source]")
st.sidebar.markdown("App developed by [Your Name]")

# Upload image
file = st.file_uploader("Choose a car photo from your computer", type=["jpg", "png"])

# Load the model
model = load_model()

# Function to make a prediction
def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Use Image.LANCZOS resampling filter
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Display results
if file is None:
    st.warning("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model)
    class_names = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    st.subheader("Prediction:")
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {prediction[0][class_index]:.2%}")

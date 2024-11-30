import streamlit as st
import tensorflow as tf
import gdown
import os

# Function to download the model from Google Drive
def download_model(model_url, model_path):
    # Check if the model is already downloaded
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    # If the model exists, but is 0 bytes, re-download
    elif os.path.getsize(model_path) == 0:  # Check if the file is empty
        print(f"Model file is empty. Re-downloading from {model_url}")
        os.remove(model_path)  # Remove the empty file
        gdown.download(model_url, model_path, quiet=False)


# Path to save the model
model_path = 'project.h5'

# URL of the model stored on Google Drive 
model_url = 'https://drive.google.com/uc?export=download&id=1axUfvqgIYoSNduZQPtz2vCFO3cmBkurK'

# Download the model if not already downloaded
download_model(model_url, model_path)

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Display the Streamlit interface
st.write("""
# Weather Classifier
""")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

# Function for image processing and prediction
def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
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
    class_names = ['shine', 'rainy', 'sunrise', 'snowy', 'cloudy']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
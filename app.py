
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import json
import requests  
from streamlit_lottie import st_lottie
import cv2
import tensorflow as tf
import numpy as np
from PIL import ImageFont

model = tf.keras.models.load_model('last_training_model.h5')

def preprocess_image(image):
    image = image.resize((50, 50))
    image = np.array(image)/255.0
    return image

def eng_arabic_mapping(predictions):
    eng_ara_mapping = {
        'aleff': 'أ',
        'bb': 'ب',
        'ta': 'ت',
        'thaa': 'ث',
        'jeem': 'ج',
        'haa': 'ح',
        'khaa': 'خ',
        'dal': 'د',
        'thal': 'ذ',
        'ra': 'ر',
        'zay': 'ز',
        'seen': 'س',
        'sheen': 'ش',
        'saad': 'ص',
        'dhad': 'ض',
        'taa': 'ط',
        'dha': 'ظ',
        'ain': 'ع',
        'ghain': 'غ',
        'fa': 'ف',
        'gaaf': 'ق',
        'kaaf': 'ك',
        'laam': 'ل',
        'meem': 'م',
        'nun': 'ن',
        'ha': 'ه',
        'waw': 'و',
        'ya': 'ي'
    }
    
    class_labels = ['aleff', 'bb', 'ta', 'thaa', 'jeem', 'haa', 'khaa', 'dal', 'thal', 'ra',
                    'zay', 'seen', 'sheen', 'saad', 'dhad', 'taa', 'dha', 'ain', 'ghain', 'fa',
                    'gaaf', 'kaaf', 'laam', 'meem', 'nun', 'ha', 'waw', 'ya']
    
    arabic_predictions = [eng_ara_mapping[label] for label in class_labels]
    predicted_arabic = [arabic_predictions[i] for i in np.argmax(predictions, axis=1)]
    
    return predicted_arabic

image2 = Image.open('/content/img.jpg')
st.image(image2)
st.title('Hmmmmmmmmm, It is translate time')

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        predictions = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_arabic = eng_arabic_mapping(predictions)
        st.write(f"Prediction: {predicted_arabic[0]}")
 

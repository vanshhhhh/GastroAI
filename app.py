import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(
    page_title="GastroAI",
    page_icon = ":medicine:"
)

@st.cache(allow_output_mutation=True)
def loadModel():
    model=load_model('assets/model/model.h5', compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model=loadModel()
st.image('assets/images/logo.png',width=300)
nav = st.radio("Navigation", ("About Us", "Disease Detection"), horizontal = True)
string = ""
if nav == "About Us":
    st.info('''
    This project is developed by [Vansh Sharma](https://www.linkedin.com/in/vanshsharma10). Make sure to give it a star on [Github](https://github.com/vanshhhhh/GastroAI).
    ''')
if nav == "Disease Detection":
    col1, col2 = st.columns(2) 
    with col1:
        file = st.file_uploader("", type=["jpg", "png"])
        def import_and_predict(image_data, model):
            size = (100,100)    
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]
            prediction = model.predict(img_reshape)
            return prediction
    with col2:
        if file is None:
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.info('Please upload an image file')
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            class_names = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
            string = "Prediction : " + class_names[np.argmax(predictions)]
            st.warning(string)

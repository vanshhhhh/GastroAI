import streamlit as st
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
nav = st.radio("Navigation", ("About", "Disease Detection"), horizontal = True)
string = ""
if nav == "About":
    st.info('''
    This project is developed by [Vansh Sharma](https://www.linkedin.com/in/vanshsharma10). Make sure to give it a star on [Github](https://github.com/vanshhhhh/GastroAI).
    ''')
    st.write('Automatic detection of diseases by use of computers is an important, but still unexplored field of research. Such innovations may improve medical practice and refine health care systems all over the world. However, datasets containing medical images are hardly available, making reproducibility and comparison of approaches almost impossible. Here, we present Kvasir, a dataset containing images from inside the gastrointestinal (GI) tract. The collection of images are classified into three important anatomical landmarks and three clinically significant findings. In addition, it contains two categories of images related to endoscopic polyp removal. Sorting and annotation of the dataset is performed by medical doctors (ex- perienced endoscopists). In this respect, Kvasir is important for research on both single- and multi-disease computer aided detec- tion. By providing it, we invite and enable multimedia researcher into the medical domain of detection and retrieval.')
    st.write('Here is the Deep Learning model used for the prediction of the disease:')
    st.image('assets/images/model_plot.png',width=500)
    st.write('The model is trained on the Kvasir dataset which contains 8 classes of images. The dataset can be found [here](https://datasets.simula.no/kvasir/).')
    st.write('Please find the model traning Jupyter Notebook here: [link](https://colab.research.google.com/github/vanshhhhh/GastroAI/blob/main/assets/notebook/model_training.ipynb)')
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
    st.error(string)

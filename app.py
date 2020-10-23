#!/usr/bin/python3
#from flask import Flask, request, jsonify, render_template
#from predict import predict_object
import streamlit as st
import time
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions


from cv2 import imdecode, IMREAD_COLOR, resize
import urllib
import numpy as np
from PIL import Image

#model = None

@st.cache(suppress_st_warning=True,show_spinner=False)
def load_model(file_path):
    with st.spinner('Loading Inceptionv3 Model...'):
        #st.write("Cache miss. Loaded model to cache")
        m = tf.keras.models.load_model(file_path,compile=False)
        return m



def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib.request.urlopen(url)
    except:
        resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = imdecode(image, IMREAD_COLOR)
    # return the image
    return image


def resize_img(img):
    img = img[...,::-1]
    img = resize(img, (299,299))
    img = img.reshape(1, 299,299,3)
    img = preprocess_input(img)
    return img


def predict_object(model,img_url):
    image = url_to_image(img_url)
    img = resize_img(image)
    #images.append(image[...,::-1])
    #print(type(model))
    result = model.predict(img)
    pred = decode_predictions(result,top=5)
    #show_image(image[...,::-1],pred[0][0][1])
    return pred[0][0][1].upper().replace("_"," ")



def get_user_input():
    url = st.sidebar.text_area("URL of an image")
    try:
        st.image(url,"User Input Image",use_column_width=True)
    except:
        pass
    return url

def predict(model,user_input):
    with st.spinner('Classifying Image...'):
        if user_input:
            prediction = predict_object(model,user_input)
            st.markdown("<h1 style='text-align: center; color: green;'>{image}</h1>".format(image=prediction), unsafe_allow_html=True)
            #print(prediction)
            if prediction:
                st.balloons()
            prediction=""

def main():

    #Header
    st.write("""
    # Image Classification using Inceptionv3
    This app classifies the image using the famous ** Inception v3 (from Google) ** which is built from scratch.     
    *Provide the url of the image to be classified in the* ***side panel***
    """)

    #Load Model
    model = load_model('Models/inceptionv3.h5')

    #Construct Side Panel
    st.sidebar.subheader("User Input")
    user_input = get_user_input()

    #Prediction
    predict(model,user_input)



if __name__ == "__main__":
    main()

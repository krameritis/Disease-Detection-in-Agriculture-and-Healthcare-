import streamlit as st 
import tensorflow as tf 
import numpy as np
import joblib as jb

st.title("Alzheimer's Diagnosis - Convolutional Neural Network(CNN)")

file=st.file_uploader("Upload Brain MRI Image")

if st.button("Upload"):
    model = jb.load("cnn_alzheimers_dementia_diagnosis_model.h5")
    test_image = tf.keras.utils.load_img(file,target_size=(176,176))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    impred = model.predict(test_image)

    def roundoff(arr):
        # To round off according to the argmax of each predicted label array.

        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr

    for classpreds in impred:
        impred = roundoff(classpreds)
    
    classcount = 1
    for count in range(4):
        if impred[count] == 1.0:
            break
        else:
            classcount+=1
    
    classdict = {1: "Mild Dementia", 2: "Moderate Dementia", 3: "No Dementia, Patient is Safe", 4: "Very Mild Dementia"}
    st.title(str(classdict[classcount]))
    

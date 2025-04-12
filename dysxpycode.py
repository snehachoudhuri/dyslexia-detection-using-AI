# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 02:46:33 2025

@author: Sneha
"""


import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Sneha/OneDrive/Desktop/minor degree/dyslexia.1', 'rb'))

# function for prediction

def dyslexia_prediction(input_data):
    
    
    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)
    print(input_data_as_numpy_array.dtype)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print(input_data_reshaped.dtype)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 1):
        print('Individual is Dyslexic. Consult a specialist.')
    elif (prediction[0]== 1):
        print('Individual is showing more severe form of Dyslexia, or a different reading difficulty. Consult a specialist immediately.')
    else:
        print('Individual is not Dyslexic')
  
def main():
    
    #title of webpg
    st.title('Dyslexia Predictor')
    
    # Use number_input for numeric values
    Language_vocab = st.number_input('Level of Language Vocabulary (range: 0 to 1)', min_value=0.0, value=0.0)
    Memory = st.number_input('Memory Level (range: 0 to 1)', min_value=0.0, value=0.0)
    Speed = st.number_input('Speed Level (range: 0 to 1)', min_value=0.0, value=0.0)
    Visual_discrimination = st.number_input('Visual Discrimination Score (range: 0 to 1)', min_value=0.0, value=0.0)
    Audio_Discrimination = st.number_input('Audio Discrimination Score (range: 0 to 1)', min_value=0.0, value=0.0)
    Survey_Score = st.number_input('Survey Score according to questionnaire (range: 0 to 1)', min_value=0.0, value=0.0)

    #code for prediction
    diagnosis = ''
    
    # Button for prediction
    if st.button('Dyslexia Test Result'):
        diagnosis = dyslexia_prediction([Language_vocab, Memory, Speed, Visual_discrimination, Audio_Discrimination, Survey_Score])
    
    st.success(diagnosis)

    
    
if __name__=='__main__':
    main()
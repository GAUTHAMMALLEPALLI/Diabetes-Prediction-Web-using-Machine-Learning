# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:36:23 2025

@author: mallepalli gautham
"""

import numpy as np
import pickle
import streamlit as st

#load the model
loaded_model=pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    
    #Changing the data into numpy array data frame
    input_data_as_numpy=np.asarray(input_data)

    #Reshaping the data
    input_data_reshaped=input_data_as_numpy.reshape(1,-1)

    #Predicting the Output
    prediction=loaded_model.predict(input_data_reshaped)

    #Printing the output as Yes or No
    if(prediction[0]==0):
      return 'Yes, The person is diabetic'
    else:
      return 'NO, The person is not diabetic'
  
def main():
    
    #Giving the title
    st.title('Diabetes prediction System by Gautham Mallepalli')
    st.image("D:/project\GAUTHAM_PRO/diabetes_image.jpg")  
    #getting the input data from the user
    Pregnancies=st.text_input('Enter Number of Pregnancies')
    Glucose=st.text_input('Enter the Glucose level')
    BloodPressure=st.text_input('Enter the Blood Pressure level')
    SkinThickness=st.text_input('Enter the Skin Thickness')
    Insulin=st.text_input('Enter the Insulin level')
    BMI=st.text_input('Enter the Body Mass Index value')
    DiabetesPedigreeFunction=st.text_input('Enter the Diabetes Pedigree Function value')
    Age=st.text_input('Enter the age')
    
    #creating a null string to store output
    diagnosis=''
    
    #creating buttion for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    #displaying output
    st.success(diagnosis)
    
if __name__=='__main__':
    main()

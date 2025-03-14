# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#load the model
loaded_model=pickle.load(open('D:/project\GAUTHAM_PRO/trained_model.sav','rb'))

#Taking the input
input_data=(3,78,50,32,88,31,0.248,26)

#Changing the data into numpy array data frame
input_data_as_numpy=np.asarray(input_data)

#Reshaping the data
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

#Predicting the Output
prediction=loaded_model.predict(input_data_reshaped)

#Printing the output as Yes or No
if(prediction[0]==0):
  print("Yes, The person is diabetic")
else:
  print("NO, The person is not diabetic")
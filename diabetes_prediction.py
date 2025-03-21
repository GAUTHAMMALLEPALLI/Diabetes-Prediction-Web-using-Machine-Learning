# -*- coding: utf-8 -*-
"""Diabetes_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CT9u_9_pEm_LIym2HFClHfu8AOnybNND

**Importing Required Libraries (Dependencies)**
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""**Data Collection and Analysis**"""

#Uploading and reading the CSV file
diabetes_dataset=pd.read_csv('/content/diabetes.csv')

#Printing first five lines of the file (optional)
diabetes_dataset.head()

#Counting no.of rows and columns (optional)
diabetes_dataset.shape

#Descriptions of values of every features (optional)
diabetes_dataset.describe()

#No.of diabetes positive people and negative people (optional)
diabetes_dataset['Outcome'].value_counts()

#Mean values of every features for the postive and negative outcomes (optional)
diabetes_dataset.groupby('Outcome').mean()

#Seperating the data and outcome
X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']

#Printing the values of X and Y after standardizing (optional)
print(X)
print(Y)

"""**Splitting, Training and Testing the data**"""

#Splitting the data into training and testing with size 80% and 20% respectively
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#Training the data using the SVM model
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

"""**Model Evaluation**"""

#Predicting the outcomes for the trained data
X_train_prediction=classifier.predict(X_train)

#Measuring accuracy score for training data
train_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(train_data_accuracy)

#Predicting the outcomes for the test data
X_test_prediction=classifier.predict(X_test)

#Measuring the accuracy score for the test data
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)

"""**Predicting System**"""

#Taking the input
input_data=(10,168,74,0,0,38,0.537,34)

#Changing the data into numpy array data frame
input_data_as_numpy=np.asarray(input_data)

#Reshaping the data
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

#Predicting the Output
prediction=classifier.predict(input_data_reshaped)

#Printing the output as Yes or No
if(prediction[0]==0):
  print("Yes, The person is diabetic")
else:
  print("NO, The person is not diabetic")

"""**Saving the Model**"""

#importing required library for saving and loading the model
import pickle

#saving the model
filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))

#loading the saved model
loaded_model=pickle.load(open('/content/trained_model.sav','rb'))

"""**Predicting using the saved model**"""

#Taking the input
input_data=(10,168,74,0,0,38,0.537,34)
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
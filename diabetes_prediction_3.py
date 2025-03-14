# -*- coding: utf-8 -*-
"""Copy of Diabetes_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15L8QT4NLJShjzlrRWUmUit-daRZKCo6D

**Importing Required Libraries (Dependencies)**
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import accuracy_score

"""**Data Collection and Analysis**"""

#Uploading and reading the CSV file
diabetes_dataset=pd.read_csv('/diabetes.csv')

"""# New Section"""

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

#Standardizing the data into same range
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)

#Reassigning the standard data to X
X = standardized_data

#Printing the values of X and Y after standardizing (optional)
print(X)
print(Y)

"""**Splitting, Training and Testing the data**"""

#Splitting the data into training and testing with size 80% and 20% respectively
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#Training the data using the Suppot Vector Machine model
classifier_svm=svm.SVC(kernel='linear')
classifier_svm.fit(X_train,Y_train)

#Training the data using the Decision Tree Model
classifier_decision_tree=tree.DecisionTreeClassifier()
classifier_decision_tree.fit(X_train,Y_train)

#Training the data using the Random Forest Model
classifier_random_forest=ensemble.RandomForestClassifier()
classifier_random_forest.fit(X_train,Y_train)

"""**Model Evaluation**"""

#Predicting the outcomes for the trained data using svm model
X_train_prediction_S=classifier_svm.predict(X_train)

#Measuring accuracy score for training data using svm model
train_data_accuracy_S=accuracy_score(X_train_prediction_S,Y_train)
print(train_data_accuracy_S)

#Predicting the outcomes for the trained data using decision tree model
X_train_prediction_D=classifier_decision_tree.predict(X_train)

#Measuring accuracy score for training data using decision tree model
train_data_accuracy_D=accuracy_score(X_train_prediction_D,Y_train)
print(train_data_accuracy_D)

#Predicting the outcomes for the trained data using random forest model
X_train_prediction_R=classifier_random_forest.predict(X_train)

#Measuring accuracy score for training data using random forest model
train_data_accuracy_R=accuracy_score(X_train_prediction_R,Y_train)
print(train_data_accuracy_R)

#Predicting the outcomes for the test data using svm model
X_test_prediction_S=classifier_svm.predict(X_test)

#Measuring the accuracy score for the test data using svm model
test_data_accuracy_S=accuracy_score(X_test_prediction_S,Y_test)
print(test_data_accuracy_S)

#Predicting the outcomes for the test data using decision tree model
X_test_prediction_D=classifier_decision_tree.predict(X_test)

#Measuring the accuracy score for the test data using svm model
test_data_accuracy_D=accuracy_score(X_test_prediction_D,Y_test)
print(test_data_accuracy_D)

#Predicting the outcomes for the test data using random forest model
X_test_prediction_R=classifier_random_forest.predict(X_test)

#Measuring the accuracy score for the test data using random forest model
test_data_accuracy_R=accuracy_score(X_test_prediction_R,Y_test)
print(test_data_accuracy_R)

"""**Predicting System**"""

#Taking the input
input_data=(10,168,74,0,0,38,0.537,34)

#Changing the data into numpy array data frame
input_data_as_numpy=np.asarray(input_data)

#Reshaping the data
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

#Standardizing the data
std_data=scaler.transform(input_data_reshaped)

#Predicting the Output
prediction=classifier_svm.predict(std_data)

#Printing the output as Yes or No
if(prediction[0]==0):
  print("Yes, The person is diabetic")
else:
  print("NO, The person is not diabetic")
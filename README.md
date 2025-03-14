**Diabetes Prediction Using Machine Learning – Web App**

**Overview**

This repository contains all the necessary files for creating a Diabetes Prediction Web Application using Machine Learning. The project involves training a predictive model on a dataset and deploying it as a web-based application.


**Project Components**


**1. Documentation**

📄 DIABETES PREDICTION USING MACHINE LEARNING TECHNIQUES.pdf

Contains theoretical background and details about the model, including the machine learning techniques used for diabetes prediction.


**2. Dataset**

📊 diabetes.csv

A dataset containing medical features and patient records used to train and evaluate the prediction model.


**3. Machine Learning Model**

🧑‍💻 diabetes prediction 3.py

Implements three different classifiers for diabetes prediction and compares their performance.

🏗 trained_model.sav

The saved model file that has been trained and used for predictions in the web application.

🔍 predictive system.py

A script to test the trained model independently without the web interface.


**4. Web Application Code**

🌐 diabetes prediction.py

The main backend script that integrates the trained model for the web app.

💻 diabetes prediction web.py

The complete web application code that loads trained_model.sav and provides an interactive interface for users to input medical details and get diabetes predictions.


**How to Use**

Train and evaluate different classifiers using diabetes prediction 3.py.

Use predictive system.py to test the trained model before web deployment.

Run diabetes prediction web.py to launch the web application.


**Requirements**

Python

Required libraries: numpy, pandas, scikit-learn, streamlit, pickle etc.


**Contribution**

Feel free to contribute by improving the classifiers, enhancing the UI, or optimizing the model.

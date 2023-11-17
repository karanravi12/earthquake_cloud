# Import necessary libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
import xgboost as xgb
import warnings
import sys

app = Flask(__name__)

# Load the model
with open('xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the user
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])
        depth = float(request.form['Depth'])

        # Create a DataFrame from the user input
        data = pd.DataFrame({'Latitude': [latitude], 'Longitude': [longitude], 'Depth': [depth]})

        # Perform prediction using the loaded model
        prediction = model.predict(data)

        # Output prediction
        return render_template('index.html', prediction=f'The predicted value is: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)

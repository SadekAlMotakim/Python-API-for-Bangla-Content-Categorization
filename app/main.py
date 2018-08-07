import pickle
from pandas.core import reshape
from flask import Flask
from flask import request
from flask import jsonify
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame


app = Flask(__name__)
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':  
        classes = ['technology', 'economy', 'entertainment', 'international', 'sports']      
        data = request.get_json()      
        lin_reg = joblib.load("LinearSVC_modelAugust.pkl")
        result =lin_reg.predict(data).tolist() 
        return jsonify(classes[result[0]])
    
@app.route("/home", methods=['POST'])
def home():
    return "Welcome"
app.run()

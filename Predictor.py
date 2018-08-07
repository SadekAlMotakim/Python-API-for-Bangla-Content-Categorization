import pickle
from pandas.core import reshape
from flask import Flask
from flask import request
from flask import jsonify
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
#app.config["DEBUG"] = True
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':  
        classes = ['technology', 'economy', 'entertainment', 'international', 'sports']      
        data = request.get_json()  
        #print(data)
        #data = DataFrame(data, index=[0], dtype=float)        
        lin_reg = joblib.load("LinearSVC_modelAugust.pkl")
        result =lin_reg.predict(data).tolist() 
        #print(classes[result[0]]);
        return jsonify(classes[result[0]])
app.run()

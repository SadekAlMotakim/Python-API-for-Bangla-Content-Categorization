from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome"

# @app.route("/predict", methods=['POST'])
# def predict():
    # if request.method == 'POST':  
    #     classes = ['technology', 'economy', 'entertainment', 'international', 'sports']      
    #     data = request.get_json()      
    #     lin_reg = joblib.load("LinearSVC_modelAugust.pkl")
    #     result =lin_reg.predict(data).tolist() 
    #     return jsonify(classes[result[0]])
if __name__ == '__main__':
  app.run()
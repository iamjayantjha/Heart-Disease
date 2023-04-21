from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model_decision_tree.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Welcome to the Heart Disease Prediction API"
@app.route('/predict/',methods=['POST'])
def predict():
    age = float(request.form.get('age'))
    sex = float(request.form.get('sex'))
    cp = float(request.form.get('cp'))
    trestbps = float(request.form.get('trestbps'))
    chol = float(request.form.get('chol'))
    fbs = float(request.form.get('fbs'))
    restecg = float(request.form.get('restecg'))
    thalach = float(request.form.get('thalach'))
    exang = float(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = float(request.form.get('slope'))
    ca = float(request.form.get('ca'))
    thal = float(request.form.get('thal'))
    input_query = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    input_data_reshaped = input_query.reshape(1,-1)
    result = model.predict(input_data_reshaped)
    return jsonify({'heart_disease':str(result)})
if __name__ == '__main__':
    
    app.run(debug=True)
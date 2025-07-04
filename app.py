from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("insurance_model.pkl", "rb"))
target_scaler = pickle.load(open("target_scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        scaled_pred = model.predict(input_data)
        predicted_charge = target_scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0][0]

        return render_template('result.html', prediction=f"${predicted_charge:.2f}")

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

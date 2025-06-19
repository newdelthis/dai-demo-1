from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("salary_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    years_exp = float(data['YearsExperience'])
    prediction = model.predict(np.array([[years_exp]]))
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

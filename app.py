import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')
    

FULL_FEATURES = [
    'family_history',
    'Growing_Stress',
    'Changes_Habits',
    'Mental_Health_History',
    'Coping_Struggles',
    'Work_Interest',
    'Social_Weakness',
    'Gender_Male',
    'self_employed_Yes',
    'mental_health_interview_Yes',
    'care_options_Yes',
    'Occupation_Business',
    'Occupation_Corporate',
    'Occupation_Housewife',
    'Occupation_Others',
    'Occupation_Student'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = {feature: 0 for feature in FULL_FEATURES}

    for feature in FULL_FEATURES:
        if feature in data:
            input_data[feature] = int(data[feature])
        else:
            print(f"Warning: Feature '{feature}' not found in input data. Using default value 0.")

    try:
        input_df = pd.DataFrame([input_data])[FULL_FEATURES]
    except Exception as e:
        return jsonify({'error': f'Error processing input data for DataFrame creation: {e}'}), 400

    prediction = model.predict(input_df)
    result = prediction[0].item()
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
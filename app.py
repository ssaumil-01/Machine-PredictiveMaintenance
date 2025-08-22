import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Load trained model
with open('predictive_maintenance.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

# List of features
FEATURES = [
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Encode product type
def encode_type(type_val):
    mapping = {'L': 0, 'M': 1, 'H': 2}
    return mapping.get(type_val, 0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get inputs from form
            type_val = request.form['Type']
            features = [
                encode_type(type_val),
                float(request.form['Air temperature [K]']),
                float(request.form['Process temperature [K]']),
                float(request.form['Rotational speed [rpm]']),
                float(request.form['Torque [Nm]']),
                float(request.form['Tool wear [min]'])
            ]
            input_df = pd.DataFrame([features], columns=FEATURES)

            # Split type vs numeric
            type_col = input_df[['Type']]
            numeric_cols = input_df.drop(columns=['Type'])
            
            # Scale only numeric features
            numeric_scaled = scaler.transform(numeric_cols)
            
            # Combine Type (unscaled) with scaled features
            final_input = np.concatenate([type_col.values, numeric_scaled], axis=1)
            
            # Model Prediction
            prediction = model.predict(input_df)[0]

            return render_template('index.html', prediction_text=f'Predicted Failure Type: {prediction}')
        except Exception as e:
            return render_template('index.html', error=f'Error: {str(e)}')
    return render_template('index.html')

if __name__ == '__main__':
    # Set debug to False in production
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
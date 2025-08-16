import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title('Predictive Maintenance Classification')

product_type=st.selectbox('Select Product Type', options=['L', 'M', 'H'])

if(product_type == 'L'):
    product_type = 0
elif(product_type == 'M'):
    product_type = 1
else:
    product_type = 2

Air_temperature= st.slider('Air temperature [K]', min_value=295.0, step=0.1,max_value=305.0)

Process_temperature=st.slider('Process temperature [K]	', min_value=305.0, step=0.1,max_value=315.0)

Rotational_speed=st.slider('Rotational speed [rpm]', min_value=1100, step=10,max_value=2900)

Torque=st.slider('Torque [Nm]', min_value=3.0, step=0.1,max_value=80.0)

Tool_wear=st.slider('Tool wear [min]', min_value=0.0, step=0.1,max_value=255.0)

def scale_features(features):
    # Load the scaler you saved during training
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler.transform([features])  

# Prepare feature vector
features = [Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear]
scaled_features = scale_features(features)

# Combine product_type with scaled features
input_data = np.hstack(([product_type], scaled_features[0]))

# Make DataFrame for model input
input_df = pd.DataFrame([input_data], columns=[
    'Type', 
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
])

if st.button('Predict'):
    with open('predictive_maintenance.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(input_df)[0]
    st.success(f'Predicted Failure Type: {prediction}')

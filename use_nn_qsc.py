#!/usr/bin/env python
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from qsc import Qsc

params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2,
          'data_location': 150
          }

def load_saved_model_and_scaler(model_path, scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data)
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return prediction

# Create the results directory
this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)

# Load the data
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
df = pd.read_csv(filename)

# Load the model and scaler
model_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}.h5")
scaler_path = os.path.join(results_path, f"nn_qsc{params['nfp']}.pkl")
model, scaler = load_saved_model_and_scaler(model_path, scaler_path)

# Create list of input data
data_array = df.iloc[params['data_location']].values
n_axis_fourier_modes = int((len(data_array)-2-8)/2)
rc = data_array[0:2*n_axis_fourier_modes:2]
zs = data_array[1:2*n_axis_fourier_modes+1:2]
eta_bar = data_array[-2-8]
B2c = data_array[-1-8]
print(f'rc = {rc}')
print(f'zs = {zs}')
print(f'eta_bar = {eta_bar}')
print(f'B2c = {B2c}')

input_X_data = data_array[:2*n_axis_fourier_modes+2]
print(f'input_X_data = {input_X_data}')
input_Y_data = data_array[2*n_axis_fourier_modes+2:]
print(f'input_Y_data = {input_Y_data}')

# Make the prediction
prediction = make_prediction(model, scaler, input_Y_data)[0]
print(f"Prediction:   {prediction}")
print(f"Actual value: {input_X_data}")
print(f'Error: {np.linalg.norm(prediction-input_X_data)}')
print(f'Relative difference %: {(prediction-input_X_data)/input_X_data*100}')
stel = Qsc(rc=rc, zs=zs, etabar=eta_bar, nfp=params['nfp'], B2c=B2c, order='r3', nphi=151)
stel.plot_boundary(r=0.1)

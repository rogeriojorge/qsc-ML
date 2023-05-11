#!/usr/bin/env python
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import load_model
from keras.optimizers import Nadam, Adam, RMSprop, SGD
from qsc import Qsc

params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2,
          'data_location': 1,
          'model': 'cnn', # 'cnn' or 'nn'
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

# Set the number of field periods
print('Usage: python use_nn_qsc.py [nfp], where nfp=2, 3 or 4. Defaults to nfp=2.')
if len(sys.argv) > 1:
    params['nfp'] = int(sys.argv[1])

# Create the results directory
this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)

# Load the data
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
df = pd.read_csv(filename)

# Load the model and scaler
model_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_model{params['model']}.h5")
scaler_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_x.pkl")
model, scaler = load_saved_model_and_scaler(model_path, scaler_path)

# Create list of input data
col_names = df.columns.tolist()
x_cols = [col for col in col_names if col.startswith('x')]
y_cols = [col for col in col_names if col.startswith('y')]
n_axis_fourier_modes = int((len(x_cols)-2)/2)
data_array = df.iloc[params['data_location']].values
rc = data_array[0:2*n_axis_fourier_modes:2]
zs = data_array[1:2*n_axis_fourier_modes+1:2]
eta_bar = data_array[n_axis_fourier_modes+1]
B2c = data_array[n_axis_fourier_modes+2]
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

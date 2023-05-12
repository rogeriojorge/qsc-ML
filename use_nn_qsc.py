#!/usr/bin/env python
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import load_model
import importlib.util

# Check if 'qsc' is installed
qsc_spec = importlib.util.find_spec('qsc')
if qsc_spec is not None:
    from qsc import Qsc
else:
    print("Module 'qsc' not found. Skipping the creation and plotting of stel.")


params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2,
          'data_location': 0,
          'model': 'nn', # 'cnn' or 'nn',
          'r_plot': 0.1,
          }

def load_saved_model_and_scaler(model_path, scaler_x_path, scaler_y_path):
    model = load_model(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_x, scaler_y

def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data)
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return prediction

# Set the number of field periods and model type
print('------------------------------------------')
print('Usage: python use_nn_qsc.py [nfp] [model], where nfp=2, 3 or 4 and model=nn or cnn. Defaults to nfp=2 and model=nn.')
print('------------------------------------------')
print('------------------------------------------')
if len(sys.argv) > 1:
    if sys.argv[1] in ['2','3','4']:
        params['nfp'] = int(sys.argv[1])
    else:
        raise ValueError('NFP must be either 2, 3 or 4')
    if len(sys.argv) > 2:
        if sys.argv[2] in ['nn', 'cnn']:
            params['model'] = sys.argv[2]
        else:
            raise ValueError('Model must be either "nn" or "cnn"')

# Create the results directory
this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)

# Load the data
# filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
# df = pd.read_csv(filename)
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')
df = pd.read_parquet(filename)

# Sort the data by L_gradB and L_gradgradB
# df = df.sort_values(by='y5', key=df['y4'].add, ascending=True)
# df = df.sort_values(by='y0', key=df['y1'].add, ascending=True)

# Load the model and scaler
model_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_model{params['model']}.h5")
scaler_x_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_x.pkl")
scaler_y_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_y.pkl")
model, scaler_x, scaler_y = load_saved_model_and_scaler(model_path, scaler_x_path, scaler_y_path)

# Create list of input data
col_names = df.columns.tolist()
x_cols = [col for col in col_names if col.startswith('x')]
y_cols = [col for col in col_names if col.startswith('y')]
n_axis_fourier_modes = int((len(x_cols)-2)/2)

# Select the data to use
df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)
df = df.sort_values(by='ysum', ascending=True)
df = df.drop(columns='ysum')
data_array = df.iloc[params['data_location']].values

input_X_data = data_array[:2*n_axis_fourier_modes+2]
input_Y_data = data_array[2*n_axis_fourier_modes+2:]

# Make the prediction
print('Making prediction...')
prediction_scaled = make_prediction(model, scaler_x, input_Y_data)
prediction = scaler_y.inverse_transform(prediction_scaled)[0]
print(f'  Error (linalg.norm) from the prediction: {np.linalg.norm(prediction-input_X_data)}')

# Print the results
rc = np.append([1],data_array[0:2*n_axis_fourier_modes:2])
predicted_rc = np.append([1],prediction[0:2*n_axis_fourier_modes:2])
zs = np.append([0],data_array[1:2*n_axis_fourier_modes+1:2])
predicted_zs = np.append([0],prediction[1:2*n_axis_fourier_modes+1:2])
eta_bar = data_array[2*n_axis_fourier_modes]
predicted_eta_bar = prediction[2*n_axis_fourier_modes]
B2c = data_array[2*n_axis_fourier_modes+1]
predicted_B2c = prediction[2*n_axis_fourier_modes+1]
np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=sys.maxsize)
print('------------------------------------------')
print(f'  true rc           = {", ".join(["{:.3e}".format(i) for i in rc])}')
print(f'  predicted_rc      = {", ".join(["{:.3e}".format(i) for i in predicted_rc])}')
print('------------------------------------------')
print(f'  true zs           = {", ".join(["{:.3e}".format(i) for i in zs])}')
print(f'  predicted_zs      = {", ".join(["{:.3e}".format(i) for i in predicted_zs])}')
print('------------------------------------------')
print(f'  true etabar       = {eta_bar:.3e}')
print(f'  predicted_etabar  = {predicted_eta_bar:.3e}')
print('------------------------------------------')
print(f'  true B2c          = {B2c:.3e}')
print(f'  predicted_B2c     = {predicted_B2c:.3e}')
print('------------------------------------------')
print('------------------------------------------')
# print(f"Prediction:   {prediction}")
# print(f"Actual value: {input_X_data}")
# print(f'Relative difference %: {(prediction-input_X_data)/input_X_data*100}')
if qsc_spec is not None:
    stel_input     = Qsc(rc=rc, zs=zs, etabar=eta_bar, nfp=int(params['nfp']), B2c=B2c, order='r3', nphi=151)
    stel_predicted = Qsc(rc=predicted_rc, zs=predicted_zs, etabar=predicted_eta_bar, nfp=int(params['nfp']), B2c=predicted_B2c, order='r3', nphi=151)
    print('Computed stel and predicted stel:')
    print('------------------------------------------')
    print(f'  true iota                  = {stel_input.iota}')
    print(f'  predicted iota             = {stel_predicted.iota}')
    print('------------------------------------------')
    print(f'  true max_elongation        = {stel_input.max_elongation}')
    print(f'  predicted max_elongation   = {stel_predicted.max_elongation}')
    print('------------------------------------------')
    print(f'  true B20_variation         = {stel_input.B20_variation}')
    print(f'  predicted B20_variation    = {stel_predicted.B20_variation}')
    print('------------------------------------------')
    print(f'  true min L_gradB           = {stel_input.min_L_grad_B}')
    print(f'  predicted min L_gradB      = {stel_predicted.min_L_grad_B}')
    print('------------------------------------------')
    print(f'  true L_gradgradB           = {stel_input.grad_grad_B_inverse_scale_length}')
    print(f'  predicted L_gradgradB      = {stel_predicted.grad_grad_B_inverse_scale_length}')
    print('------------------------------------------')
    print(f'  true r_singularity         = {stel_input.r_singularity}')
    print(f'  predicted r_singularity    = {stel_predicted.r_singularity}')
    print('------------------------------------------')
    print(f'  true d2_volume_d_psi2      = {stel_input.d2_volume_d_psi2}')
    print(f'  predicted d2_volume_d_psi2 = {stel_predicted.d2_volume_d_psi2}')
    # stel.plot_boundary(r=params['r_plot'])

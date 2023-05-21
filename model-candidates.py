import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qsc import Qsc

params = {
    'results_path': 'results',
    'data_path': 'data',
    'nfp': 2,
    'n_samples': 5000,
    'gmm_n_components': 19,
    'model': 'nn',
}

def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data)
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    prediction = scaler_y.inverse_transform(prediction)[0]
    return prediction

this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')
df = pd.read_parquet(filename).astype('float64').sample(params['n_samples'])
y_columns = [col for col in df.columns if col.startswith('y')]
Y = df[y_columns]

train_data, test_data = train_test_split(Y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('gmm', GaussianMixture(random_state=1337))
])
pipeline.set_params(gmm__n_components=params['gmm_n_components'])
pipeline.fit(train_data)

# Prepare a list to hold the samples
samples = []

number_of_samples = 10000

# Continue sampling until we have 10k samples with values > 0
while len(samples) < number_of_samples:
    # Sample a batch of 1000 samples at once for efficiency
    batch = pipeline.steps[0][1].inverse_transform(pipeline.steps[1][1].sample(10000)[0])
    # Filter out samples with values <= 0
    batch = batch = batch[np.all(batch > 0, axis=1)]
    # Append the valid samples to our list
    samples.extend(batch)

# Truncate the list to exactly 10k samples
samples = samples[:number_of_samples]

# Convert the list to a numpy array for convenience
samples = np.array(samples)

model_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_model{params['model']}.h5")
scaler_x_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_scaler_x.pkl")
scaler_y_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_scaler_y.pkl")

model = load_model(model_path)
scaler_x = joblib.load(scaler_x_path)
scaler_y = joblib.load(scaler_y_path)

n_samples_index = 0
prediction = make_prediction(model, scaler_x, samples[n_samples_index])

n_axis_fourier_modes = int((len(prediction)-2)/2)
predicted_rc = np.append([1],prediction[0:2*n_axis_fourier_modes:2])
predicted_zs = np.append([0],prediction[1:2*n_axis_fourier_modes+1:2])
predicted_eta_bar = prediction[2*n_axis_fourier_modes]
predicted_B2c = prediction[2*n_axis_fourier_modes+1]
stel = Qsc(rc=predicted_rc, zs=predicted_zs, etabar=predicted_eta_bar, nfp=int(params['nfp']), B2c=predicted_B2c, order='r3', nphi=151)
print('------------------------------------------')
print(f'  predicted_rc      = {", ".join(["{:.3e}".format(i) for i in predicted_rc])}')
print(f'  predicted_zs      = {", ".join(["{:.3e}".format(i) for i in predicted_zs])}')
print(f'  predicted_etabar  = {predicted_eta_bar:.3e}')
print(f'  predicted_B2c     = {predicted_B2c:.3e}')
print('------------------------------------------')
print(f'  predicted_iota          = {stel.iota:.3e}')
print(f'  predicted_elongation    = {stel.max_elongation:.3e}')
print(f'  predicted_L_gradB       = {stel.min_L_grad_B:.3e}')
print(f'  predicted_L_gradgradB   = {1/stel.grad_grad_B_inverse_scale_length:.3e}')
print(f'  predicted_B20_variation = {stel.B20_variation:.3e}')
print(f'  predicted_r_singularity = {stel.r_singularity:.3e}')

stel.plot_boundary(r=0.1)
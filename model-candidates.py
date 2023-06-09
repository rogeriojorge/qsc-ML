import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    'nfp': 3,
    'n_samples': 1000,
    'gmm_n_components': 19,
    'model': 'nn',
    'number_of_samples': 200,
}

def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data)
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    prediction = scaler_y.inverse_transform(prediction)[0]
    return prediction

this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)
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

# Continue sampling until we have number_of_samples samples with values > 0
while len(samples) < params['number_of_samples']:
    # Sample a batch of 1000 samples at once for efficiency
    batch = pipeline.steps[0][1].inverse_transform(pipeline.steps[1][1].sample(params['number_of_samples'])[0])
    # Filter out samples with values <= 0
    ########## Why was this filter here?? ##########
    # batch = batch = batch[np.all(batch > 0, axis=1)]
    # Append the valid samples to our list
    samples.extend(batch)

# Truncate the list to exactly number_of_samples samples
samples = np.array(samples[:params['number_of_samples']])

model_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_model{params['model']}.h5")
scaler_x_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_scaler_x.pkl")
scaler_y_path = os.path.join(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_scaler_y.pkl")

model = load_model(model_path)
scaler_x = joblib.load(scaler_x_path)
scaler_y = joblib.load(scaler_y_path)

n_samples_index = 0
sample = samples[n_samples_index]
prediction = make_prediction(model, scaler_x, sample)

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
print(f'  predicted_iota          = {np.abs(stel.iota):.3f}')
print(f'  asked_iota              = {sample[0]:.3f}')
print('---------------------')
print(f'  predicted_r_singularity = {stel.r_singularity:.3f}')
print(f'  asked_r_singularity     = {sample[1]:.3f}')
print('---------------------')
print(f'  predicted_B20_variation = {stel.B20_variation:.3f}')
print(f'  asked_B20_variation     = {1/sample[2]:.3f}')
print('---------------------')
print(f'  predicted_L_gradB       = {stel.min_L_grad_B:.3f}')
print(f'  asked_L_gradB           = {sample[3]:.3f}')
print('---------------------')
print(f'  predicted_L_gradgradB   = {1/stel.grad_grad_B_inverse_scale_length:.3f}')
print(f'  asked_L_gradgradB       = {sample[4]:.3f}')
print('---------------------')
print(f'  predicted_elongation    = {stel.max_elongation:.3f}')
print(f'  asked_elongation        = {1/sample[5]:.3f}')
print('---------------------')
print(f'  predicted_d2_volume_dpsi = {stel.d2_volume_d_psi2:.3f}')

# stel.plot_boundary(r=stel.r_singularity)

samples_scaled = scaler_x.transform(samples)
predictions_scaled = model.predict(samples_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# f'y0': 0.33*np.abs(1/iota),
# f'y1': 0.06/r_singularity,
# f'y2': B20_variation,
# f'y3': elongation/8,
# f'y4': 0.6/L_grad_B,
# f'y5': 0.6/L_grad_grad_B,
# f'y6': 0.3/min_R0,

Y_test = []
filtered_samples = []
for count, prediction_sample in tqdm(enumerate(predictions), total=len(predictions), desc="Processing"):
    rc = np.append([1],prediction_sample[0:2*n_axis_fourier_modes:2])
    zs = np.append([0],prediction_sample[1:2*n_axis_fourier_modes+1:2])
    eta_bar = prediction_sample[2*n_axis_fourier_modes]
    B2c = prediction_sample[2*n_axis_fourier_modes+1]
    stel = Qsc(rc=rc, zs=zs, etabar=eta_bar, nfp=int(params['nfp']), B2c=B2c, order='r2', nphi=71)
        # f'y0': np.abs(iota),#0.33*np.abs(1/iota),
        # f'y1': r_singularity,#0.06/r_singularity,
        # f'y2': 1/B20_variation,#B20_variation,
        # f'y3': L_grad_B,#0.6/L_grad_B,
        # f'y4': L_grad_grad_B,#0.6/L_grad_grad_B,
        # f'y5': 1/min_R0,#0.3/min_R0,
        # f'y6': 1/elongation,#elongation/8,
    array_to_append = np.array([
        np.abs(stel.iota),
        stel.r_singularity,
        1/stel.B20_variation,
        stel.min_L_grad_B,
        1/stel.grad_grad_B_inverse_scale_length,
        stel.min_R0,
        1/stel.max_elongation])
    diff = np.abs((array_to_append-samples[count])/samples[count])
    if len(diff[diff>1])>0:
        # print("--------------------------------------------------")
        # print(f"Skipping sample {count} because of large difference between predicted and asked parameters.")
    # if any(abs(param) > 100 for param in array_to_append) or any(abs(sample) > 100 for sample in samples[count]):
    #     print("--------------------------------------------------")
    #     print(f"Skipping sample {count} because of large parameter value.")
    #     print(f"Parameters: {array_to_append}")
    #     print(f"Samples: {samples[count]}")
        continue
    Y_test.append(array_to_append)
    filtered_samples.append(samples[count])
Y_test = np.array(Y_test)

# loss, mae = model.evaluate(predictions, Y_test, verbose=0)
# print(f"Test MAE: {mae}")
# print(f"Test Loss: {loss}")
print(f"Fraction of correct predictions: {len(Y_test)/len(predictions)}")

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, filtered_samples, label='Predicted values', alpha=0.5)
min_x = np.min([np.min(Y_test), np.min(filtered_samples)])
max_x = np.max([np.max(Y_test), np.max(filtered_samples)])
plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'GaussianMixture nfp={params["nfp"]}, model={params["model"]}, gmm_n_components={params["gmm_n_components"]}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"GaussianMixture_predictions_qsc_nfp{params['nfp']}_model{params['model']}.png"))

#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

from keras.models import load_model
params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2,
          'model': 'nn',
          'random_state': 42,
          'test_size': 0.2,
          'n_data_subset': 650000,
          'n_best': 100000,  # number of best predictions to retain
          }

def load_saved_model_and_scaler(model_path, scaler_x_path, scaler_y_path):
    model = load_model(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_x, scaler_y

def preprocess_data(X, Y, scaler_x, scaler_y):
    X = scaler_x.transform(X)
    Y = scaler_y.transform(Y)
    return X, Y

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

# Load the model and scaler
model_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_model{params['model']}.h5")
scaler_x_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_x.pkl")
scaler_y_path = os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_y.pkl")
model, scaler_x, scaler_y = load_saved_model_and_scaler(model_path, scaler_x_path, scaler_y_path)

# Load the data
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')
df = pd.read_parquet(filename).sample(params['n_data_subset'],random_state=params['random_state'])
for column in df.columns:
    if df[column].dtype.byteorder == '>':
        df[column] = df[column].values.byteswap().newbyteorder()
x_columns = [col for col in df.columns if col.startswith('x')]
y_columns = [col for col in df.columns if col.startswith('y')]
Y = df[x_columns].values
X = df[y_columns].values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'], random_state=params['random_state'])
# X_train, X_test, Y_train, Y_test = preprocess_data(X_train, X_test, Y_train, Y_test, scaler_x, scaler_y)
X_scaled, Y_scaled = preprocess_data(X, Y, scaler_x, scaler_y)

loss, mae = model.evaluate(X_scaled, Y_scaled, verbose=0)
print(f"Test MAE: {mae}")
print(f"Test Loss: {loss}")

predictions = model(X_scaled)
Y_scaled_descaled =  Y_scaled#scaler_y.inverse_transform(Y_scaled)
predictions_descaled = predictions#scaler_y.inverse_transform(predictions)

#####
relative_errors = np.abs((predictions_descaled - Y_scaled_descaled)/predictions_descaled)
worst_indices = np.argmax(relative_errors, axis=1)
worst_errors = relative_errors[np.arange(len(predictions_descaled)), worst_indices]
worst_predicted_indices_count = np.bincount(worst_indices, minlength=len(predictions_descaled[0]))
plt.figure(figsize=(10,5))
plt.bar(range(len(worst_predicted_indices_count)), worst_predicted_indices_count)
plt.xlabel('Index')
plt.ylabel('Number of times worst predicted')
plt.title('Histogram of Worst Predicted Indices')
plt.savefig(os.path.join(results_path, f"nn_worst_prediction_indices_qsc_nfp{params['nfp']}_model{params['model']}.png"))

plt.figure(figsize=(8, 8))
plt.scatter(Y_scaled, predictions, label='Predicted values', alpha=0.5)
min_x = np.min([np.min(Y_scaled), np.min(predictions)])
max_x = np.max([np.max(Y_scaled), np.max(predictions)])
plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'nfp={params["nfp"]}, model={params["model"]}, metric={mae:.3f}, loss={loss:.3f}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"nn_predictions_all_qsc_nfp{params['nfp']}_model{params['model']}.png"))

# Create a DataFrame to hold original data and associated errors
relative_errors_1 = np.abs((predictions_descaled - Y_scaled_descaled)/predictions_descaled)
relative_errors_2 = np.abs((predictions_descaled - Y_scaled_descaled)/Y_scaled_descaled)

df_with_errors = df.copy()
df_with_errors['error_1'] = np.max(relative_errors_1, axis=1)
df_sorted = df_with_errors.sort_values(by=['error_1'])
df_best = df_sorted.head(2*params['n_best'])
df_with_errors['error_2'] = np.max(relative_errors_2, axis=1)
df_sorted = df_with_errors.sort_values(by=['error_2'])
df_best = df_sorted.head(params['n_best'])
df_best = df_best.drop(columns=['error_1'])
df_best = df_best.drop(columns=['error_2'])

best_filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}_best.parquet')
df_best.to_parquet(best_filename)

# Load the best data
best_filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}_best.parquet')
df_best = pd.read_parquet(best_filename)

Y_best = df_best[x_columns].values
X_best = df_best[y_columns].values

X_best_scaled, Y_best_scaled = preprocess_data(X_best, Y_best, scaler_x, scaler_y)

best_predictions = model(X_best_scaled)

best_loss, best_mae = model.evaluate(X_best_scaled, Y_best_scaled, verbose=0)
print(f"Best MAE: {best_mae}")
print(f"Best Loss: {best_loss}")

plt.figure(figsize=(8, 8))
plt.scatter(Y_best_scaled, best_predictions, label='Predicted values', alpha=0.5)
min_x = np.min([np.min(Y_best_scaled), np.min(best_predictions)])
max_x = np.max([np.max(Y_best_scaled), np.max(best_predictions)])
plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'nfp={params["nfp"]}, model={params["model"]}, Best metric={best_mae:.3f}, Best loss={best_loss:.3f}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"nn_predictions_best_qsc_nfp{params['nfp']}_model{params['model']}.png"))

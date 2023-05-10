#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import HeNormal
from keras.layers import BatchNormalization, LeakyReLU
from keras.optimizers.legacy import Nadam, Adam, RMSprop, SGD
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers.schedules import ExponentialDecay
from keras.losses import Huber
import tensorflow as tf
import joblib

params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2,
          'model': 'cnn',# 'cnn' or 'nn',
          'optimizer': Adam,# 'Adam', 'Nadam', 'RMSprop', 'SGD',
          'learning_rate': 0.006,
          'epochs': 50,
          'batch_size': 512,
          }

# Function to build the neural network
def build_neural_network(input_shape, output_shape):
    model = Sequential([
        Dropout(0.5),
        Dense(512, kernel_initializer=HeNormal(), input_shape=(input_shape,)),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, kernel_initializer=HeNormal()),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        # Dense(128, kernel_initializer=HeNormal()),
        # LeakyReLU(),
        # BatchNormalization(),
        # Dropout(0.1),
        Dense(64, kernel_initializer=HeNormal()),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(output_shape)
    ])
    return model

# Function to build the 1D CNN
def build_cnn(input_shape, output_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(output_shape)
    ])
    return model

# Set the number of field periods
print('Usage: python nn_qsc.py [nfp], where nfp=2, 3 or 4. Defaults to nfp=2.')
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

# Split the data into training and test sets
x_columns = [col for col in df.columns if col.startswith('x')]
y_columns = [col for col in df.columns if col.startswith('y')]

## ACTUALLY SOLVING THE INVERSE PROBLEM
# X = df[x_columns].values
# Y = df[y_columns].values
Y = df[x_columns].values
X = df[y_columns].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the data
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# Build the neural network
if params['model'] == 'cnn':
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_shape = (X_train.shape[1], 1)
    output_shape = Y_train.shape[1]
    model = build_cnn(input_shape, output_shape)
else:
    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]
    model = build_neural_network(input_shape, output_shape)

# Train the model
learning_rate = ExponentialDecay(initial_learning_rate=params['learning_rate'], decay_steps=1000, decay_rate=0.9)
metrics = ['mae']
model.compile(optimizer=params['optimizer'](learning_rate=learning_rate), loss=Huber(), metrics=metrics)
model.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_split=0.2, verbose=1)

# loss, mae, rmse, mape, r2, cosine_similarity, log_cosh_error = model.evaluate(X_test, Y_test, verbose=0)
loss, metric = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test metric: {metric}")
print(f"Test loss: {loss}")

# Save the model and scaler to files
model.save(os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}.h5"))
joblib.dump(scaler_x, os.path.join(results_path, f"nn_qsc{params['nfp']}.pkl"))

# Make predictions
predictions = model(X_test)

# Plot the results
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, predictions, label='Predicted values', alpha=0.5)
min_x = np.min([Y_test, predictions])
max_x = np.max([Y_test, predictions])
plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')  # Adjust the range according to your data
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'nfp={params["nfp"]}, model={params["model"]}, metric={metric:.3f}, loss={loss:.3f}, epochs={params["epochs"]}, batch_size={params["batch_size"]}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"nn_qsc_plot{params['nfp']}_model{params['model']}.png"))
plt.show()
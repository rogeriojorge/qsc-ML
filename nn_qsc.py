#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
import joblib

params = {'results_path': 'results',
          'data_path': 'data',
          'nfp': 2}

# Function to build the neural network
def build_neural_network(input_shape, output_shape):
    model = Sequential([
        Dense(512, kernel_initializer=HeNormal(), input_shape=(input_shape,)),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, kernel_initializer=HeNormal()),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, kernel_initializer=HeNormal()),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, kernel_initializer=HeNormal()),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.1),
        Dense(output_shape)
    ])

    model.compile(optimizer=Nadam(learning_rate=0.001), loss='mse', metrics=['mae'])
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
input_shape = X_train.shape[1]
output_shape = Y_train.shape[1]

model = build_neural_network(input_shape, output_shape)

# Train the model
epochs = 200  # Increased number of epochs
batch_size = 64
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MAE: {mae}")

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
plt.title(f'True values vs Predicted values (nfp={params["nfp"]}))')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"nn_qsc_plot{params['nfp']}.png"))
plt.show()
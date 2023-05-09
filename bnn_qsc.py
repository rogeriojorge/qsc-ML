#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
exit()
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

params = {
    'results_path': 'results',
    'data_path': 'data',
}

this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)
nfp =  int(sys.argv[1])
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{nfp}.csv')
df = pd.read_csv(filename)
# Extract X and Y columns
x_columns = [col for col in df.columns if col.startswith('x')]
y_columns = [col for col in df.columns if col.startswith('y')]

X = df[x_columns].values
Y = df[y_columns].values

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the input features
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

def build_bayesian_neural_network(input_shape, output_shape):
    # Define the prior weight distribution
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n, dtype=dtype), scale_diag=tf.ones(n, dtype=dtype))

    # Define the posterior weight distribution
    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype)

    # Define the Bayesian neural network
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tfp.layers.DenseVariational(128, activation='relu', make_prior_fn=prior, make_posterior_fn=posterior, kl_weight=1/X_train.shape[0]),
        tfp.layers.DenseVariational(64, activation='relu', make_prior_fn=prior, make_posterior_fn=posterior, kl_weight=1/X_train.shape[0]),
        tfp.layers.DenseVariational(output_shape, make_prior_fn=prior, make_posterior_fn=posterior, kl_weight=1/X_train.shape[0])
    ])

    return model

input_shape = X_train.shape[1]
output_shape = Y_train.shape[1]

model = build_bayesian_neural_network(input_shape, output_shape)

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MAE: {mae}")

# Make predictions
predictions = model(X_test)

predicted_mean = predictions.mean()
predicted_stddev = predictions.stddev()

predicted_mean = scaler_x.inverse_transform(predicted_mean)
predicted_stddev = scaler_x.inverse_transform(predicted_stddev)

confidence_level = 0.95
lower_bound = predicted_mean - (predicted_stddev * confidence_level)
upper_bound = predicted_mean + (predicted_stddev * confidence_level)

# Plot the first x value predictions and their confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(Y_test[:, 0], label='True values')
plt.plot(predicted_mean[:, 0], label='Predicted values')
plt.fill_between(range(len(Y_test)), lower_bound[:, 0], upper_bound[:, 0], color='gray', alpha=0.5, label='Confidence interval')
plt.xlabel('Samples')
plt.ylabel('X1')
plt.legend()
plt.show()

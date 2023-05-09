#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp

def build_bayesian_neural_network(input_shape, output_shape):
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(
                loc=t[..., :n],
                scale=1e-5 + tf.nn.softplus(c + t[..., n:])))
        ])

    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=1))
        ])

    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(
            units=128,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1/X_train.shape[0],
            activation='relu',
            input_shape=(input_shape,)),
        tfp.layers.DenseVariational(
            units=64,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1/X_train.shape[0],
            activation='relu'),
        tfp.layers.DenseVariational(
            units=output_shape,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1/X_train.shape[0])
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

params = {'results_path': 'results', 'data_path': 'data', 'nfp': 2}
if len(sys.argv) > 1:
    params['nfp'] = int(sys.argv[1])

this_path = str(Path(__file__).parent.resolve())
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
df = pd.read_csv(filename)

x_columns = [col for col in df.columns if col.startswith('x')]
y_columns = [col for col in df.columns if col.startswith('y')]

X = df[x_columns].values
Y = df[y_columns].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

input_shape = X_train.shape[1]
output_shape = Y_train.shape[1]

model = build_bayesian_neural_network(input_shape, output_shape)

# Train the model
epochs = 100
batch_size = 32
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MAE: {mae}")

predictions = model(X_test)
predicted_mean = predictions

confidence_level = 0.95  # 95% confidence interval with a normal distribution
lower_bound = predicted_mean - (mae * confidence_level)
upper_bound = predicted_mean + (mae * confidence_level)

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, predicted_mean, label='Predicted values', alpha=0.5)
plt.plot([0, 2], [0, 2], 'r', label='Perfect predictions')  # Adjust the range according to your data

plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True values vs Predicted values')
plt.legend()
plt.grid(True)
plt.show()

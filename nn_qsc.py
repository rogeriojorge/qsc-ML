#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import (Dense, Dropout, Conv1D,  Flatten, Input, BatchNormalization)
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Sequential, Model
import joblib

params = {
    # Change this parameter to select the amount of data to use
    'n_data_subset': 400000,
    'results_path': 'results',
    'data_path': 'data',
    'nfp': 3,
    'model': 'nn', # 'cnn' or 'nn'
    'optimizer': Adam,
    'learning_rate': 0.004,
    'epochs': 400,
    'batch_size': 2048,
    'early_stopping_patience': 30,
    'test_size': 0.2,
    'random_state': 42,
    'reg_strength': 1e-5,
    'dropout_rate': 1e-3,
    'validation_split': 0.2,
    'decay_steps': 1000,
    'decay_rate': 0.9,
    'train_with_best_points': False,
    'use_even_better_points': False,
    'select_best_sum_y': False,
    'use_sum_y_as_X': False,
    'columns_to_use': 'all'#['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']
}

def build_neural_network(input_shape, output_shape, reg_strength=1e-7, dropout_rate=3e-3):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(reg_strength)),
        # BatchNormalization(),
        Dropout(dropout_rate),
        # Dense(1024, activation='relu', kernel_regularizer=l2(reg_strength)),
        # # BatchNormalization(),
        # Dropout(dropout_rate),
        # Dense(512, activation='relu', kernel_regularizer=l2(reg_strength)),
        # # BatchNormalization(),
        # Dropout(dropout_rate),
        Dense(256, activation='relu', kernel_regularizer=l2(reg_strength)),
        # BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=l2(reg_strength)),
        # BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(reg_strength)),
        # BatchNormalization(),
        Dropout(dropout_rate),
        Dense(output_shape)
    ])
    return model

def build_cnn(input_shape, output_shape, reg_strength=1e-7, dropout_rate=3e-3):
    input_layer = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(reg_strength))(input_layer)
    # x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(reg_strength))(x)
    # x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(reg_strength))(x)
    # x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = Dropout(dropout_rate)(x)

    output_layer = Dense(output_shape, activation='linear')(x)

    return Model(input_layer, output_layer)

def preprocess_data(X_train, X_test, Y_train, Y_test, params):
    scaler_x = StandardScaler().fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler().fit(Y_train)
    Y_train = scaler_y.transform(Y_train)
    Y_test = scaler_y.transform(Y_test)

    if params['model'] == 'cnn':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        input_shape = (X_train.shape[1], 1)
    else:
        input_shape = X_train.shape[1]

    output_shape = Y_train.shape[1]

    return X_train, X_test, input_shape, output_shape, scaler_x, Y_train, Y_test, scaler_y

def create_dataset(X_train, Y_train, params):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(params['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset

# Set the number of field periods and model type
print('Usage: python use_nn_qsc.py [nfp] [model], where nfp=2, 3 or 4 and model=nn or cnn. Defaults to nfp=2 and model=nn.')
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

this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)

# filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
# df = pd.read_csv(filename)
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}')
if params['train_with_best_points']:
    filename+="_best"
if params['use_even_better_points']:
    filename+="_best"
df = pd.read_parquet(filename+".parquet")
if params['n_data_subset'] > len(df):
    print(f'Warning: n_data_subset {params["n_data_subset"]} is larger than the number of data points available {len(df)}. Using all data points.')
    params['n_data_subset'] = len(df)
else:
    if params['select_best_sum_y']:
        df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)
        df = df.sort_values(by='ysum', ascending=True).head(params['n_data_subset'])
        df = df.drop(columns='ysum')
    else:
        df = df.sample(params['n_data_subset'],random_state=params['random_state'])

for column in df.columns:
    if df[column].dtype.byteorder == '>':
        df[column] = df[column].values.byteswap().newbyteorder()
# df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)
# df = df.sort_values(by='ysum', ascending=True).head(params['n_data_subset'])
# df = df.drop(columns='ysum')

x_columns = [col for col in df.columns if col.startswith('x')]
y_columns = params['columns_to_use'] if params['columns_to_use'] != 'all' else [col for col in df.columns if col.startswith('y')]
#y = [iota,   r_singularity, 1/B20_variation, L_grad_B, L_grad_grad_B, min_R0, 1/elongation]

## ACTUALLY SOLVING THE INVERSE PROBLEM
Y = df[x_columns].values
if params['use_sum_y_as_X']:
    df['summed_y'] = df[y_columns].sum(axis=1)
    X = df['summed_y'].values.reshape(-1, 1)  # reshape is required because this is a single feature
else:
    X = df[y_columns].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'], random_state=params['random_state'])

X_train, X_test, input_shape, output_shape, scaler_x, Y_train, Y_test, scaler_y = preprocess_data(X_train, X_test, Y_train, Y_test, params)

if params['model'] == 'cnn':
    model = build_cnn(input_shape, output_shape, reg_strength=params['reg_strength'], dropout_rate=params['dropout_rate'])
else:
    model = build_neural_network(input_shape, output_shape, reg_strength=params['reg_strength'], dropout_rate=params['dropout_rate'])

# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=params['learning_rate'], decay_steps=params['decay_steps'], decay_rate=params['decay_rate']
# )

optimizer = params['optimizer'](learning_rate=params['learning_rate'])#learning_rate)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=params['early_stopping_patience'], restore_best_weights=True)

train_dataset = create_dataset(X_train, Y_train, params)
validation_dataset = create_dataset(X_test, Y_test, params)

history = model.fit(train_dataset, epochs=params['epochs'], validation_data=validation_dataset,
          validation_split=params['validation_split'], verbose=1, callbacks=[early_stopping])

loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MAE: {mae}")
print(f"Test Loss: {loss}")

predictions = model(X_test)

# Save the model and scaler
model.save(os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_model{params['model']}"+(f"_best" if params['train_with_best_points'] else "")+".h5"))
joblib.dump(scaler_x, os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_x"+(f"_best" if params['train_with_best_points'] else "")+".pkl"))
joblib.dump(scaler_y, os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_scaler_y"+(f"_best" if params['train_with_best_points'] else "")+".pkl"))
print(f"Model and scaler saved in {results_path}")

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, predictions, label='Predicted values', alpha=0.5)
min_x = np.min([np.min(Y_test), np.min(predictions)])
max_x = np.max([np.max(Y_test), np.max(predictions)])
plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'nfp={params["nfp"]}, model={params["model"]}, metric={mae:.3f}, loss={loss:.3f}, epochs={params["epochs"]}, batch_size={params["batch_size"]}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, f"nn_predictions_qsc_nfp{params['nfp']}_model{params['model']}"+(f"_best" if params['train_with_best_points'] else "")+".png"))

plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.savefig(os.path.join(results_path, f"nn_history_qsc_nfp{params['nfp']}_model{params['model']}"+(f"_best" if params['train_with_best_points'] else "")+".png"))

plt.show()
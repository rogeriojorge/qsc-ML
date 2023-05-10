#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

params = {
    'results_path': 'results',
    'data_path': 'data',
    'nfp': 2,
    'model': 'cnn',
    'optimizer': Adam,
    'learning_rate': 0.006,
    'epochs': 50,
    'batch_size': 512,
    'early_stopping_patience': 10,
}

def build_neural_network(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Dropout(0.5)(inputs)
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_shape)(x)
    return Model(inputs=inputs, outputs=outputs)

def build_cnn(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_shape)(x)
    return Model(inputs=inputs, outputs=outputs)

def main():
    if len(sys.argv) > 1:
        params['nfp'] = int(sys.argv[1])

    this_path = str(Path(__file__).parent.resolve())
    general_results_path = os.path.join(this_path, params['results_path'])
    results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
    os.makedirs(results_path, exist_ok=True)

    filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
    df = pd.read_csv(filename)

    x_columns = [col for col in df.columns if col.startswith('x')]
    y_columns = [col for col in df.columns if col.startswith('y')]

    Y = df[x_columns].values
    X = df[y_columns].values

    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    metrics_list = []
    losses_list = []

    for fold, (train_indices, test_indices) in enumerate(kfold.split(X, Y)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        scaler_x = StandardScaler().fit(X_train)
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)

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

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params['learning_rate'], decay_steps=1000, decay_rate=0.9
        )

        model.compile(optimizer=params['optimizer'](learning_rate=learning_rate), loss=Huber(), metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=params['early_stopping_patience'], restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_fold{fold + 1}_best.h5"), monitor='val_loss', save_best_only=True)

        model.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_split=0.2, verbose=1, callbacks=[early_stopping, model_checkpoint])

        loss, metric = model.evaluate(X_test, Y_test, verbose=0)
        metrics_list.append(metric)
        losses_list.append(loss)

        predictions = model(X_test)

        plt.figure(figsize=(8, 8))
        plt.scatter(Y_test, predictions, label='Predicted values', alpha=0.5)
        min_x = np.min([Y_test, predictions])
        max_x = np.max([Y_test, predictions])
        plt.plot([min_x, max_x], [min_x, max_x], 'r', label='Perfect predictions')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title(f'Fold {fold + 1}, nfp={params["nfp"]}, model={params["model"]}, metric={metric:.3f}, loss={loss:.3f}, epochs={params["epochs"]}, batch_size={params["batch_size"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_path, f"nn_qsc_fold{fold + 1}_plot{params['nfp']}_model{params['model']}.png"))
        plt.show()

    avg_metric = np.mean(metrics_list)
    avg_loss = np.mean(losses_list)
    print(f"Average metric: {avg_metric}")
    print(f"Average loss: {avg_loss}")
    # Save the average metric and loss to a text file
    with open(os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}_summary.txt"), 'w') as summary_file:
        summary_file.write(f"Average metric: {avg_metric}\n")
        summary_file.write(f"Average loss: {avg_loss}\n")

    # Save the model and scaler to files
    model.save(os.path.join(results_path, f"nn_qsc_nfp{params['nfp']}.h5"))
    joblib.dump(scaler_x, os.path.join(results_path, f"nn_qsc{params['nfp']}.pkl"))
    
if __name__ == 'main':
    main()
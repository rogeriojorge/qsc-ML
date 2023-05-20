import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.special import kl_div
from keras.models import load_model
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
import joblib
import matplotlib.pyplot as plt
from scipy.stats import entropy


params = {
    'results_path': 'results',
    'data_path': 'data',
    'nfp': 2,
    'test_size': 0.2,
    'random_state': 42,
    'n_components': 5,
    'n_samples': 1000,
    'model': 'nn',
    'n_data_subset': 3000,
    'n_samples_subset': 400,
}


def get_path(name):
    return Path(params['results_path'], f"nfp{params['nfp']}", f"nn_qsc_nfp{params['nfp']}_{name}")


def load_data(path: str):
    df = pd.read_parquet(path).apply(pd.to_numeric, downcast='float', errors='ignore')
    df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)
    df = df.nsmallest(params['n_data_subset'], 'ysum').drop(columns='ysum')
    return train_test_split(df.filter(like='y').values, df.filter(like='x').values, test_size=params['test_size'], random_state=params['random_state'], shuffle=True)


def calculate_kl(p, q):
    return entropy(p, q)


def fit_gmm(data):
    return GaussianMixture(n_components=params['n_components']).fit(data)


def get_samples(data, num_samples):
    return np.random.choice(data.flatten(), num_samples)


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def plot_graphs(bins, data_subset, samples_subset, kl_divergences):
    kde_data = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(data_subset[:, None])
    logprob_data = kde_data.score_samples(bins[:, None])

    kde_samples = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(samples_subset[:, None])
    logprob_samples = kde_samples.score_samples(bins[:, None])

    gmm_data = GaussianMixture(n_components=2).fit(data_subset[:, None])
    logprob_gmm_data = gmm_data.score_samples(bins[:, None])

    gmm_samples = GaussianMixture(n_components=2).fit(samples_subset[:, None])
    logprob_gmm_samples = gmm_samples.score_samples(bins[:, None])

    freq_data, bin_edges_data = np.histogram(data_subset, bins=bins, density=True)
    freq_samples, bin_edges_samples = np.histogram(samples_subset, bins=bins, density=True)
    
    midpoints_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2
    midpoints_samples = (bin_edges_samples[:-1] + bin_edges_samples[1:]) / 2

    plt.figure(figsize=(12, 8))

    plt.subplot(311)
    plt.plot(midpoints_data, freq_data, label='Data')
    plt.plot(midpoints_samples, freq_samples, label='Samples')
    plt.title(f'Histogram (KL divergence = {kl_divergences[0]:.2f})')
    plt.legend()

    plt.subplot(312)
    plt.plot(bins, np.exp(logprob_data), label='Data')
    plt.plot(bins, np.exp(logprob_samples), label='Samples')
    plt.title(f'KDE (KL divergence = {kl_divergences[1]:.2f})')
    plt.legend()

    plt.subplot(313)
    plt.plot(bins, np.exp(logprob_gmm_data), label='Data')
    plt.plot(bins, np.exp(logprob_gmm_samples), label='Samples')
    plt.title(f'GMM (KL divergence = {kl_divergences[2]:.2f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    this_path = os.path.abspath('')
    filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')
    X_train, X_test, Y_train, Y_test = load_data(filename)
    
    gmm_all = fit_gmm(np.concatenate((X_train, Y_train), axis=1))
    gmm_input = fit_gmm(X_train)
    
    model = load_model(get_path(f'model{params["model"]}.h5'))
    scaler_x = joblib.load(get_path('scaler_x.pkl'))
    scaler_y = joblib.load(get_path('scaler_y.pkl'))

    samples_all = gmm_all.sample(params['n_samples'])[0][:, :X_train.shape[1]]
    samples_input = gmm_input.sample(params['n_samples'])[0]
    
    predictions_all = model.predict(samples_all)
    predictions_input = model.predict(samples_input)

    # Ensure the shapes of predictions and Y_test subsets match
    if predictions_all.shape[0] != Y_test.shape[0]:
        predictions_all = predictions_all[:Y_test.shape[0], :]
    if predictions_input.shape[0] != Y_test.shape[0]:
        predictions_input = predictions_input[:Y_test.shape[0], :]
    
    Y_test_subset_all = Y_test[:predictions_all.shape[0], :]
    mae_all = calculate_mae(scaler_y.inverse_transform(predictions_all), Y_test_subset_all)

    Y_test_subset_input = Y_test[:predictions_input.shape[0], :]
    mae_input = calculate_mae(scaler_y.inverse_transform(predictions_input), Y_test_subset_input)

    bins = np.linspace(-10, 10, 100)
    kl_divergences_all = []
    kl_divergences_input = []

    for idx in range(X_train.shape[1]):
        data_subset_all = get_samples(Y_test[:, idx], params['n_samples_subset'])
        samples_subset_all = predictions_all[:params['n_samples_subset'], idx]
        kl_divergence_all = calculate_kl(data_subset_all, samples_subset_all)
        kl_divergences_all.append(kl_divergence_all)
        
        data_subset_input = get_samples(Y_test[:, idx], params['n_samples_subset'])
        samples_subset_input = predictions_input[:params['n_samples_subset'], idx]
        kl_divergence_input = calculate_kl(data_subset_input, samples_subset_input)
        kl_divergences_input.append(kl_divergence_input)

    plot_graphs(bins, data_subset_all, samples_subset_all, kl_divergences_all)
    plot_graphs(bins, data_subset_input, samples_subset_input, kl_divergences_input)

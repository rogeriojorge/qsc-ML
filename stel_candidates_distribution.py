import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
from keras.models import load_model
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import zscore
from scipy.stats import iqr

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

def optimal_bandwidth(data):
    return 1.06 * np.std(data) * len(data)**(-1/5)

def calculate_kl(p, q):
    return entropy(p, q)

def fit_gmm(data):
    return GaussianMixture(n_components=params['n_components']).fit(data)

def get_samples(data, num_samples):
    return np.random.choice(data.flatten(), num_samples)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def remove_outliers(X, Y):
    z_scores = zscore(X)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return X[filtered_entries], Y[filtered_entries]

def plot_graphs(bins, data_subset, samples_subset, kl_divergences, results_path, plot_name):
    bandwidth_data = optimal_bandwidth(data_subset)
    bandwidth_samples = optimal_bandwidth(samples_subset)

    kde_data = KernelDensity(bandwidth=bandwidth_data, kernel='gaussian').fit(data_subset[:, None])
    kde_samples = KernelDensity(bandwidth=bandwidth_samples, kernel='gaussian').fit(samples_subset[:, None])

    logprob_data = kde_data.score_samples(bins[:, None])
    logprob_samples = kde_samples.score_samples(bins[:, None])

    gmm_data = GaussianMixture(n_components=2).fit(data_subset[:, None])
    logprob_gmm_data = gmm_data.score_samples(bins[:, None])

    gmm_samples = GaussianMixture(n_components=2).fit(samples_subset[:, None])
    logprob_gmm_samples = gmm_samples.score_samples(bins[:, None])

    freq_data, bin_edges_data = np.histogram(data_subset, bins=bins, density=True)
    freq_samples, bin_edges_samples = np.histogram(samples_subset, bins=bins, density=True)
    
    midpoints_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2
    midpoints_samples = (bin_edges_samples[:-1] + bin_edges_samples[1:]) / 2

    plt.figure(figsize=(5, 8))

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
    plt.savefig(os.path.join(results_path, f'{plot_name}.png'), dpi=200)
    plt.show()

if __name__ == "__main__":
    this_path = os.path.abspath('')
    general_results_path = os.path.join(this_path, params['results_path'])
    results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
    os.makedirs(results_path, exist_ok=True)
    filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')

    X_train, X_test, Y_train, Y_test = load_data(filename)
    X_train, Y_train = remove_outliers(X_train, Y_train)
    X_test, Y_test   = remove_outliers(X_test, Y_test)

    model = load_model(get_path(f'model{params["model"]}.h5'))
    scaler_x = joblib.load(get_path('scaler_x.pkl'))
    scaler_y = joblib.load(get_path('scaler_y.pkl'))

    predictions_test = model.predict(X_test)
    mae_test = calculate_mae(scaler_y.inverse_transform(predictions_test), Y_test)
    print(f'MAE for test set: {mae_test:.2f}')

    gmm_input = fit_gmm(X_train)
    samples_input = gmm_input.sample(params['n_samples'])[0]
    predictions_input = model.predict(samples_input)

    min_val = min(np.min(Y_test), np.min(predictions_input))
    max_val = max(np.max(Y_test), np.max(predictions_input))

    margin = 0.05 * (max_val - min_val)  # 5% of the range
    bins = np.linspace(min_val - margin, max_val + margin, 100)

    kl_divergences_input = []

    for idx in range(X_train.shape[1]):
        data_subset_input = get_samples(Y_test[:, idx], params['n_samples_subset'])
        samples_subset_input = predictions_input[:params['n_samples_subset'], idx]

        p_data, _ = np.histogram(data_subset_input, bins=bins, density=True)
        p_samples, _ = np.histogram(samples_subset_input, bins=bins, density=True)

        kl_divergence_input = calculate_kl(p_data, p_samples)
        kl_divergences_input.append(kl_divergence_input)

    print('Finished KL divergence calculations.')
    
    plot_graphs(bins, data_subset_input, samples_subset_input, kl_divergences_input, results_path, 'kl_input_results')
    print('Plotted and saved figures for input results.')
    print('Finished all calculations and visualizations.')


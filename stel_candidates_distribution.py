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
    return train_test_split(df.filter(like='y').values, df.filter(like='x').values, test_size=params['test_size'], random_state=params['random_state'])


def calculate_kl(p, q):
    return entropy(p, q)


def fit_gmm(data):
    return GaussianMixture(n_components=params['n_components']).fit(data)


def get_samples(data, num_samples):
    return np.random.choice(data.flatten(), num_samples)


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def plot_graphs(bins, data_subset, samples_subset, kl_divergences):
    hist_data, _ = np.histogram(data_subset, bins=bins, density=True)
    hist_samples, _ = np.histogram(samples_subset, bins=bins, density=True)

    kde_data = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(data_subset[:, None])
    logprob_data = kde_data.score_samples(bins[:, None])

    kde_samples = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(samples_subset[:, None])
    logprob_samples = kde_samples.score_samples(bins[:, None])

    gmm_data = GaussianMixture(n_components=2).fit(data_subset[:, None])
    logprob_gmm_data = gmm_data.score_samples(bins[:, None])

    gmm_samples = GaussianMixture(n_components=2).fit(samples_subset[:, None])
    logprob_gmm_samples = gmm_samples.score_samples(bins[:, None])

    plt.figure(figsize=(12, 8))

    plt.subplot(311)
    plt.plot(bins, hist_data, label='Data')
    plt.plot(bins, hist_samples, label='Samples')
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

    samples_all = gmm_all.sample(params['n_samples'])[0]
    samples_input = gmm_input.sample(len(X_train))[0]
    
    predictions_all = model.predict(samples_all)
    predictions_input = model.predict(samples_input)
    
    mae_all = calculate_mae(Y_test, predictions_all)
    mae_input = calculate_mae(Y_test, predictions_input)

    data_subset = get_samples(np.concatenate((X_train, Y_train), axis=1), params['n_samples_subset'])
    samples_subset = get_samples(samples_all, params['n_samples_subset'])

    bins = np.linspace(min(data_subset.min(), samples_subset.min()), 
                       max(data_subset.max(), samples_subset.max()), 100)

    hist_data, _ = np.histogram(data_subset, bins=bins, density=True)
    hist_samples, _ = np.histogram(samples_subset, bins=bins, density=True)

    kl_div_hist = calculate_kl(hist_data+1e-10, hist_samples+1e-10)
    kl_div_kde = calculate_kl(np.exp(logprob_data), np.exp(logprob_samples))
    kl_div_gmm = calculate_kl(np.exp(logprob_gmm_data), np.exp(logprob_gmm_samples))

    kl_divergences = [kl_div_hist, kl_div_kde, kl_div_gmm]

    plot_graphs(bins, data_subset, samples_subset, kl_divergences)

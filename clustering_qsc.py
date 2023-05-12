#!/usr/bin/env python
import os, sys, numpy as np, pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

params = {
    'results_path': 'results',
    'data_path': 'data',
    'train_test_split': 0.2,
    'nfp': 2,
    'n_data_subset': 3000,
    'n_estimators': 300,
    'min_clusters': 3,
    'max_clusters': 10,
    'regression': 'multivariate', # 'multivariate' or 'univariate,
}

print('Usage: python tse_qsc.py [nfp], where nfp=2, 3 or 4. Defaults to nfp=2.')
if len(sys.argv) > 1:
    if sys.argv[1] in ['2','3','4']:
        params['nfp'] = int(sys.argv[1])
    else:
        raise ValueError('NFP must be either 2, 3 or 4')

this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
results_path = os.path.join(general_results_path, f'nfp{params["nfp"]}')
os.makedirs(results_path, exist_ok=True)
os.chdir(this_path)
# filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.csv')
# df = pd.read_csv(filename)
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{params["nfp"]}.parquet')
df = pd.read_parquet(filename)
# Only use a subset of parameters
for column in df.columns:
    if df[column].dtype.byteorder == '>':
        df[column] = df[column].values.byteswap().newbyteorder()
df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)
df = df.sort_values(by='ysum', ascending=True).head(params['n_data_subset'])
df = df.drop(columns='ysum')

if params['regression'] == 'multivariate':
    x_cols = [col for col in df.columns if col.startswith('x')]
    y_cols = [col for col in df.columns if col.startswith('y')]
    X = StandardScaler().fit_transform(df[x_cols])
    y = df[y_cols]
elif params['regression'] == 'univariate':
    df['y'] = df.filter(like='y', axis=1).sum(axis=1)
    rel_cols = [col for col in df.columns if col.startswith('x') or col == 'y']
    df_rel = df.loc[:, rel_cols]
    X = StandardScaler().fit_transform(df_rel.drop(columns=['y']))
    y = df_rel['y']

print('Finding X, y and splitting into training and test sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['train_test_split'], random_state=42)

# Use a subset of the data for clustering and semi-supervised learning
X_subset = X_train#[:params['n_data_subset']]
y_subset = y_train#[:params['n_data_subset']]

print('Random forest regression...')

# Initialize a RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=params['n_estimators']).fit(X_subset, y_subset)

print('Determining the optimal number of clusters using the elbow method...')
wcss = []
n_clusters_search = np.linspace(params['min_clusters'], params['max_clusters'], params['max_clusters']-params['min_clusters']+1, dtype=int)
for n_cluster in n_clusters_search:
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_subset)
    wcss.append(kmeans.inertia_)
    print(f'  {n_cluster} clusters of {params["max_clusters"]} has wcss = {kmeans.inertia_}')
plt.plot(n_clusters_search, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig(os.path.join(results_path, f"elbow_method_nclusters_nfp{params['nfp']}.png"))
plt.close()

n_clusters = n_clusters_search[np.argmin(wcss)]
print('  Optimum number of clusters: ', n_clusters, ' (WCSS = ', min(wcss), ')')
print(f'Fitting KMeans with the optimal number of clusters...')
kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(X_subset)
labels = kmeans.labels_

print('Fitting the semi-supervised learning model...')
label_propagation_model = LabelSpreading(kernel='rbf', alpha=0.8, max_iter=200)
label_propagation_model.fit(X_subset, labels)
y_test_pred = rfc.predict(X_test)
y_test_semi_supervised = label_propagation_model.predict(X_test)

# Predicting labels using the LabelSpreading model
semi_supervised_labels = label_propagation_model.predict(X_subset).tolist()

print('Reducing dimensionality with t-SNE for plotting...')
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_subset)

print('Saving models and predictions...')

# with open(os.path.join(results_path,'rfc_model.pkl'), 'wb') as f:
#     pickle.dump(rfc, f)
with open(os.path.join(results_path,f'label_propagation_model_nfp{params["nfp"]}.pkl'), 'wb') as f:
    pickle.dump(label_propagation_model, f)

np.savetxt(os.path.join(results_path, f'y_test_pred_nfp{params["nfp"]}.csv'), y_test_pred, delimiter=',')
np.savetxt(os.path.join(results_path, f'y_test_semi_supervised_nfp{params["nfp"]}.csv'), y_test_semi_supervised, delimiter=',')

print('Predicting labels...')
predicted_labels = kmeans.predict(X_subset).tolist()

print('Plotting Kmeans clusters...')
# Prepare the data for plotting
x_values = X_tsne[:, 0]
y_values = X_tsne[:, 1]

# Create a scatter plot
plt.figure(figsize=(10, 7))
# Plot the scatter points for the predicted labels
plt.scatter(x_values, y_values, c=predicted_labels, cmap='viridis')
# Calculate the cluster centers and plot them
unique_labels = np.unique(predicted_labels)
cluster_centers_2D = np.array([X_tsne[predicted_labels == label].mean(axis=0) for label in unique_labels])
plt.scatter(cluster_centers_2D[:, 0], cluster_centers_2D[:, 1], c='red', s=300, alpha=0.7)
plt.title('t-SNE with K-Means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(os.path.join(results_path, f"tsne_nclusters{n_clusters}_nfp{params['nfp']}.png"))

# qsc-ML

The `qsc-ML` repository aims to create a framework for applying various machine learning techniques to stellarator optimization. Stellarator optimization involves finding good surfaces and coils that can reproduce the magnetic field on those surfaces. This repository simplifies the optimization problem by employing the near-axis expansion formalism, which reduces the parameter space from hundreds to around a dozen.

The code was created using Python3.11 but should be able to run with Python>=3.8.

As of now the repository is able to perform the following tasks:

- Loading near-axis (QSC) data
- Dimensionality Reduction using t-SNE
- Neural Network based Regression
- Cluster Analysis
- Model Application for Predictions

The repository contains five main files:

1. `load_qsc_data.py`: This script generates CSV files from the qsc code output. When run, it creates a folder called "data" with the CSV files.
2. `tsne_qsc.py`: This script uses t-SNE and HDBSCAN to map the parameter space to a lower dimensional space and identify clusters. It also plots the resulting stellarators based on the pyQSC package.
3. `nn_qsc.py`: This is a draft attempt to use neural networks to obtain a representation of the inverse problem (given a desired set of stellarator properties, find the corresponding near-axis parameters).
4. `use_nn_qsc.py`: This is an example usage of the neural network trained with `nn_qsc.py` to find new stellarators.
5. `clustering_qsc.py`: Use clustering methods to analyze the data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Loading Data](#loading-data)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Neural Network Regression](#neural-network-regression)
  - [Applying the Neural Network Model](#applying-the-neural-network-model)
  - [Cluster Analysis](#cluster-analysis)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/rogeriojorge/qsc-ML.git
cd qsc-ML
pip install -r requirements.txt
```

---

## Dependencies

The script requires the packages that are listed in the `requirements.txt` file.
To install them all, run the following command in the root folder of the repository
```bash
pip install -r requirements.txt
```

---


## Usage

### Loading Data

If you want to use new data, you can use the script `load_qsc_data.py` to load the data. Otherwise, you can use the data that is already present in the repository.
This script will load a CSV file containing the dataset and convert it into a pandas dataframe. It also takes care of any big-endian to little-endian conversion.

```bash
python load_qsc_data.py
```

The data must come from an output of the `qsc` code. Link: https://github.com/landreman/qsc
The script creates a `data` directory with the subdirectory `raw_data`, which contains copies of the input QSC output files, as well as the generated CSV files.

For each input file, the script generates a corresponding CSV file with the same basename, i.e., if the input file is `qsc_out.random_scan_nfp2.nc`, the output CSV file will be named `qsc_out.random_scan_nfp2.csv`.

The CSV files contain the following columns:

- `x1`, `x2`, ..., `xN`: The input parameters of the QSC, including the R0c and Z0s values.
- `y0`: The inverse rotational transform (scaled by 0.33).
- `y1`: The inverse distance to the magnetic axis singularity (scaled by 0.06).
- `y2`: The B20 variation.
- `y3`: The elongation (scaled by 8).
- `y4`: The inverse magnetic field gradient length (scaled by 0.6).
- `y5`: The inverse second magnetic field gradient length (scaled by 0.6).
- `y6`: The inverse minimum major radius (scaled by 0.3).


### Dimensionality Reduction

The `tsne_qsc.py` script is used for visualizing high-dimensional data. It uses t-SNE to reduce the dimensionality of the data and visualize it in a 2D or 3D plot.

```bash
python tsne_qsc.py
```

The tsne_qsc.py script performs dimensionality reduction on the QSC data. High-dimensional data is challenging to visualize and understand. This script uses the t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm to reduce the dimensionality of the data down to two or three dimensions.

t-SNE is a non-linear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. It maps multi-dimensional data to two or more dimensions suitable for human observation.

This script also uses the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm to identify clusters in the reduced dimensionality data. The identified clusters are then visualized using the pyQSC package.

The output of this script is a visualization of the data in reduced dimensions, with different clusters shown in different colors. This can help in understanding the underlying structure and patterns in the data.

### Neural Network Regression

Use the `nn_qsc.py` script to create a neural network model using TensorFlow and Keras. The script will also train this model on your data and save it for future use.

```bash
python nn_qsc.py
```

The nn_qsc.py script is used to create and train a neural network model on the QSC data. Neural networks are a powerful tool for regression and classification problems, and in this case, they are used for regression.

The script first prepares the data by standardizing the features (subtracting the mean and dividing by the standard deviation). It then creates a neural network model using the TensorFlow and Keras libraries. The model is a fully connected (dense) network, and the architecture (number of layers and neurons per layer) can be adjusted in the script.

The model is then trained on the prepared data. Training a neural network involves feeding the data into the network, comparing the network's prediction with the true output, and updating the network's weights based on the difference (the error). This process is repeated multiple times (epochs), and each time the model's performance should improve.

The output of this script is the trained neural network model, which is saved to a file. This model can then be used to make predictions on new data.

### Applying the Neural Network Model

The `use_nn_qsc.py` script is used to apply the trained neural network model on new data. The script will load the model and the new data, make predictions, and save these predictions in a CSV file.

```bash
python use_nn_qsc.py
```

The use_nn_qsc.py script is used to apply the trained neural network model to new data. The script loads the model and the new data, uses the model to make predictions on the new data, and saves these predictions to a CSV file.

The predictions made by the model can be used in various ways. For example, they can be used to identify promising areas of the parameter space to explore in more detail, to identify outliers or anomalies in the data, or to simulate the behavior of the system under different conditions.

### Cluster Analysis

The `clustering_qsc.py` script uses K-Means clustering and semi-supervised learning to analyze the data. It also determines the optimal number of clusters and saves the models and predictions for future use.

```bash
python clustering_qsc.py
```

The clustering_qsc.py script is responsible for conducting an unsupervised learning task on the data, namely clustering. The purpose of this step is to discover inherent groupings within the data which may represent different classes or types of stellarator configurations.

This script uses the K-Means algorithm, a popular clustering technique that partitions the data into K distinct, non-overlapping clusters. Each data point belongs to the cluster with the nearest mean value. The script also utilizes a method for determining the optimal number of clusters to avoid overfitting or underfitting the data.

Moreover, this script employs semi-supervised learning techniques. Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. This can be highly valuable in situations where labeling data is costly or time-consuming.

After the clustering, the script saves the models and the prediction results for each data point, which identifies the cluster it belongs to.

This script offers a way to understand the high-level structure of the dataset and to identify potential trends or patterns. It can also help in identifying outlier data points that do not fit well into any of the identified clusters.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgements

Thanks to Gonçalo Abreu for kickstarting this project.

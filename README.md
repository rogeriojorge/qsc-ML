# qsc-ML

The `qsc-ML` repository aims to create a framework for applying various machine learning techniques to stellarator optimization. Stellarator optimization involves finding good surfaces and coils that can reproduce the magnetic field on those surfaces. This repository simplifies the optimization problem by employing the near-axis expansion formalism, which reduces the parameter space from hundreds to around a dozen.

The repository contains four main files:

1. `load_qsc_data.py`: This script generates CSV files from the qsc code output. When run, it creates a folder called "data" with the CSV files.
2. `tsne_qsc.py`: This script uses t-SNE and HDBSCAN to map the parameter space to a lower dimensional space and identify clusters. It also plots the resulting stellarators based on the pyQSC package.
3. `nn_qsc.py`: This is a draft attempt to use neural networks to obtain a representation of the inverse problem (given a desired set of stellarator properties, find the corresponding near-axis parameters).
4. `use_nn_qsc.py`: This is a draft attempt to use Bayesian neural networks to obtain a representation of the inverse problem (given a desired set of stellarator properties, find the corresponding near-axis parameters).

# 1. load_qsc_data.py

`load_qsc_data.py` is a Python script that converts the Quasisymmetric Stellarator Configurations (QSC) output data from NetCDF format to CSV files. The script also organizes the data in a structured format that can be easily loaded into other scripts for further analysis and processing.

## Usage

```bash
python load_qsc_data.py [filename]
```

Where `filename` is the path to the QSC output file in NetCDF format. The script also supports processing multiple QSC output files at once. If no filename is provided, the script will default to process the following files:

- `qsc_out.random_scan_nfp2.nc`
- `qsc_out.random_scan_nfp3.nc`
- `qsc_out.random_scan_nfp4.nc`

## Dependencies

The script requires the following packages:

- numpy
- pandas
- pathlib
- scipy
- shutil

## Output

The script creates a `data` directory with two subdirectories:

1. `raw_data`: This directory contains copies of the input QSC output files.
2. `data`: This directory contains the generated CSV files.

For each input file, the script generates a corresponding CSV file with the same basename, i.e., if the input file is `qsc_out.random_scan_nfp2.nc`, the output CSV file will be named `qsc_out.random_scan_nfp2.csv`.

The CSV files contain the following columns:

- `x1`, `x2`, ..., `xN`: The input parameters of the QSC, including the R0c and Z0s values.
- `y0`: The inverse rotational transform (scaled by 0.33).
- `y1`: The inverse distance to the magnetic axis singularity (scaled by 0.09).
- `y2`: The negative inverse second derivative of the volume with respect to the magnetic flux (scaled by -80).
- `y3`: The B20 variation (scaled by 1.1).
- `y4`: The elongation (scaled by 8).
- `y5`: The inverse magnetic field gradient length (scaled by 0.3).
- `y6`: The inverse second magnetic field gradient length (scaled by 0.3).
- `y7`: The inverse minimum major radius (scaled by 0.3).

## Example

To convert a QSC output file `qsc_out.example.nc` to a CSV file, run the following command:

```bash
python load_qsc_data.py qsc_out.example.nc
```

The script will generate a CSV file named `qsc_out.example.csv` in the `data` directory.

# 2. tsne_qsc.py

`tsne_qsc.py` is a Python script that performs t-Distributed Stochastic Neighbor Embedding (t-SNE) and HDBSCAN clustering on data generated from Quasisymmetric Stellarator Configurations (QSC). The script also generates and saves 2D and 3D visualizations of the clustering results.

### Usage

```bash
python tsne_qsc.py [nfp] [dimensionality]
```

Where `nfp` is the number of field periods (2, 3, or 4) and `dimensionality` is the number of dimensions for t-SNE (2 or 3). The script defaults to `nfp=2` and `dimensionality=2`.

### Dependencies

The script requires the following packages:

- numpy
- pandas
- pathlib
- matplotlib
- mpl_toolkits
- PIL (Pillow)
- bhtsne
- sklearn
- hdbscan
- bokeh
- qsc

### Configuration

You can customize various parameters for t-SNE, HDBSCAN, and QSC by modifying the `params` dictionary in the script.

### Output

The script saves the following files in the `results` directory:

- t-SNE scatter plot image (`tse_nfp{nfp}_perplexity{perplexity}_{tsne_parameters}params.png`)
- HDBSCAN clustering Bokeh plot (`bokeh_tse_stellarators_nfp{nfp}_perplexity{perplexity}.html` and `bokeh_tse_stellarators_nfp{nfp}_perplexity{perplexity}.png`) for 2D t-SNE
- HDBSCAN clustering Plotly plot (`plotly_tse_stellarators_nfp{nfp}_perplexity{perplexity}_3parameters.html` and `plotly_tse_stellarators_nfp{nfp}_perplexity{perplexity}_3parameters.png`) for 3D t-SNE

# 3. nn_qsc.py

The `nn_qsc.py` script trains a neural network to solve the inverse problem for Quasisymmetric Stellarators. The neural network is implemented using TensorFlow and Keras. The main features of the script are:

- Loading data from a CSV file containing the random scans of the Quasisymmetric Stellarators.
- Preprocessing the data by splitting it into training and test sets, and scaling the input data.
- Building and training a neural network model using the preprocessed data.
- Evaluating the performance of the trained model on the test data.
- Saving the trained model and the scaler for later use.
- Generating plots of the true values vs predicted values to visualize the model's performance.

### Usage

To run the `nn_qsc.py` script, simply execute the following command in the terminal:

```bash
python nn_qsc.py [nfp]
```

where `nfp` is the number of field periods (2, 3, or 4). The default value is 2.

# 4. use_nn_qsc.py

The `use_nn_qsc.py` script demonstrates how to use the trained neural network model from the `nn_qsc.py` script to make predictions. The main features of the script are:

- Loading the saved neural network model and scaler from the results directory.
- Loading data from a CSV file containing the random scans of the Quasisymmetric Stellarators.
- Preparing input data for the neural network by selecting specific data points from the loaded data.
- Using the loaded model and scaler to make a prediction based on the prepared input data.
- Comparing the predicted values with the actual values and calculating the error.
- Visualizing the stellarator boundary using the Qsc Python library.

### Usage

To run the `use_nn_qsc.py` script, execute the following command in the terminal:

```bash
python use_nn_qsc.py [nfp]
```

Note that this script uses the trained model and scaler from the `nn_qsc.py` script, so make sure to run `nn_qsc.py` first to generate the necessary files.

## Dependencies

The `nn_qsc.py` and `use_nn_qsc.py` scripts require the following packages:

- numpy
- pandas
- pathlib
- matplotlib
- scikit-learn
- tensorflow
- keras
- qsc

## Acknowledgements

Thanks to Gonçalo Abreu for kickstarting this project.
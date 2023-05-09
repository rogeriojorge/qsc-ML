The `qsc-ML` repository aims to create a framework for applying various machine learning techniques to stellarator optimization. Stellarator optimization involves finding good surfaces and coils that can reproduce the magnetic field on those surfaces. This repository simplifies the optimization problem by employing the near-axis expansion formalism, which reduces the parameter space from hundreds to around a dozen.

The repository contains three main files:

1. `load_qsc_data.py`: This script generates CSV files from the qsc code output. When run, it creates a folder called "data" with the CSV files.
2. `tse_qsc.py`: This script uses t-SNE and HDBSCAN to map the parameter space to a lower dimensional space and identify clusters. It also plots the resulting stellarators based on the pyQSC package.
3. `nn_qsc.py`: This is a draft attempt to use neural networks to obtain a representation of the inverse problem (given a desired set of stellarator properties, find the corresponding near-axis parameters).
4. `bnn_qsc.py`: This is a draft attempt to use Bayesian neural networks to obtain a representation of the inverse problem (given a desired set of stellarator properties, find the corresponding near-axis parameters).

The `tse_qsc.py` script is a Python script used for visualizing and clustering the dataset of near-axis configurations using t-SNE and HDBSCAN. The script performs the following tasks:

1. Reads a CSV file containing the dataset of near-axis configurations.
2. Applies t-SNE for dimensionality reduction and visualizes the results using matplotlib.
3. Performs HDBSCAN clustering and uses Bokeh to create an interactive plot with the identified clusters (if 2D t-SNE is used) or uses Plotly for visualization (if 3D t-SNE is used).
4. Includes a function called `create_3d_image` that generates a 3D image of a stellarator given its parameters using matplotlib. It returns a base64-encoded version of the image that can be used in the Bokeh plot as an ImageURL glyph.

The script uses several libraries, including numpy, pandas, matplotlib, bhtsne, sklearn, hdbscan, bokeh, and qsc from the qsc library. The script takes two optional arguments, `nfp` (number of field periods) and `dimensionality`.

In the code snippet you provided, the script reads the parameters and checks if additional arguments are passed. If so, it updates the parameters accordingly. The `create_3d_image` function is defined, which generates a 3D image of a stellarator given its parameters. The script then proceeds to load the CSV data, perform t-SNE and HDBSCAN, plot the results, and create the interactive plot with Bokeh (if 2D t-SNE is used) or Plotly (if 3D t-SNE is used).


Thanks to Gonçalo Abreu for kickstarting this project.
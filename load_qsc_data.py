#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.io import netcdf_file
import pandas as pd
from pathlib import Path
import shutil
this_path = str(Path(__file__).parent.resolve())
data_path = os.path.join(this_path, 'data')
raw_data_path = os.path.join(data_path, 'raw_data')
os.makedirs(data_path, exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)

save_as_float32 = True
n_data_to_keep = 700000

if len(sys.argv) == 1:
    nfps = [2,3,4]
    filenames  = [f'/Users/rogeriojorge//local/qsc/examples/qsc_out.random_scan_nfp{nfp}.nc' for nfp in nfps]
    qsc_inputs = [f'/Users/rogeriojorge//local/qsc/examples/qsc_in.random_scan_nfp{nfp}' for nfp in nfps]
    [shutil.copyfile(qsc_input, os.path.join(raw_data_path, os.path.basename(qsc_input))) for qsc_input in qsc_inputs]
else:
   filenames = [sys.argv[1]]

for filename in filenames:
    bare_filename = os.path.basename(filename)
    shutil.copyfile(filename, os.path.join(raw_data_path, bare_filename))
    s = 'qsc_out.'
    if bare_filename[:len(s)] != s or filename[-3:] != '.nc':
        raise RuntimeError("A qsc_out.*.nc file must be provided as an argument")

    f = netcdf_file(filename, 'r', mmap=False)
    n_scan = f.variables['n_scan'][()]
    iota = np.abs(f.variables['scan_iota'][()])
    eta_bar = f.variables['scan_eta_bar'][()]
    B2c = f.variables['scan_B2c'][()]
    B20_variation = f.variables['scan_B20_variation'][()]
    r_singularity = f.variables['scan_r_singularity'][()]
    d2_volume_d_psi2 = f.variables['scan_d2_volume_d_psi2'][()]
    L_grad_B = f.variables['scan_min_L_grad_B'][()]
    L_grad_grad_B = f.variables['scan_min_L_grad_grad_B'][()]
    elongation = f.variables['scan_max_elongation'][()]
    min_R0 = f.variables['scan_min_R0'][()]
    R0c = f.variables['scan_R0c'][()]
    Z0s = f.variables['scan_Z0s'][()]

    # For r_singularity, replace 1e30 with 1
    r_singularity = np.minimum(r_singularity, np.ones_like(r_singularity))

    # Create a DataFrame from the extracted variables
    data = {}

    # Add each column of R0c and Z0s to the data dictionary
    for i in range(1,R0c.shape[1]): #assume R0c0=1 and Z0s0=0
        data[f'x{2*i-1}'] = R0c[:, i]
        data[f'x{2*i}'] = Z0s[:, i]

    data.update({
        f'x{2*R0c.shape[1]-1}': eta_bar,
        f'x{2*R0c.shape[1]}': B2c,
        f'y0': np.abs(iota),#0.33*np.abs(1/iota),
        f'y1': r_singularity,#0.06/r_singularity,
        f'y2': 1/B20_variation,#B20_variation,
        f'y3': L_grad_B,#0.6/L_grad_B,
        f'y4': L_grad_grad_B,#0.6/L_grad_grad_B,
        f'y5': min_R0,#0.3/min_R0,
        f'y6': 1/elongation,#elongation/8,
        # f'y7': d2_volume_d_psi2,#-80/d2_volume_d_psi2,
    })

    df = pd.DataFrame(data).sample(n_data_to_keep,random_state=42)

    # Ensure data is in the correct byte order
    for column in df.columns:
        if df[column].dtype.byteorder == '>':
            df[column] = df[column].values.byteswap().newbyteorder()

    # # Create a new column that is the sum of all y columns
    # df['ysum'] = df.loc[:, df.columns.str.startswith('y')].sum(axis=1)

    # # Sort by this new column and keep only the top n_data_to_keep rows
    # df = df.sort_values(by='ysum', ascending=True).head(n_data_to_keep)

    # # Drop the 'ysum' column as it's no longer needed
    # df = df.drop(columns='ysum')

    # Find all columns that are of type float64
    float64_cols = df.select_dtypes(include=['float64']).columns

    # Cast those columns to float32
    if save_as_float32:
        df[float64_cols] = df[float64_cols].astype('float32')

    # # Save the DataFrame to a CSV file
    # csv_filename = os.path.join(data_path, str(Path(filename).stem) + '.csv')
    # df.to_csv(csv_filename, index=False)
    # print(f"CSV file created: {csv_filename}")

    # Save the DataFrame to a Parquet file
    parquet_filename = os.path.join(data_path, str(Path(filename).stem) + '.parquet')
    df.to_parquet(parquet_filename, compression='BROTLI', index=False)
    print(f"Parquet file created: {parquet_filename} with {len(df)} rows")
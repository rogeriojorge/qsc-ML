#!/usr/bin/env python
import numpy as np
from qsc import Qsc
import matplotlib.pyplot as plt
r_singularity=[]
B20_variation=[]
B2c_array = np.linspace(-10, 10, 150)
for B2c in B2c_array:
    stel = Qsc(rc=[1, 0.018367347], zs=[0, -0.036734694], nfp=4, etabar=-0.864285714, B2c=B2c, nphi=91, order='r2')
    r_singularity.append(stel.r_singularity)
    B20_variation.append(stel.B20_variation)
print('min r_singularity = ', np.min(r_singularity))
print('min B20_variation = ', np.min(B20_variation))
location_min_r_singularity = np.argmin(r_singularity)
location_min_B20_variation = np.argmin(B20_variation)
stel = Qsc(rc=[1, 0.018367347], zs=[0, -0.036734694], nfp=4, etabar=-0.864285714, B2c=B2c_array[location_min_r_singularity], nphi=91, order='r2')
stel.plot_boundary()
plt.plot(B2c_array, r_singularity/np.max(r_singularity), label='r_singularity')
plt.plot(B2c_array, B20_variation/np.max(B20_variation), label='B20_variation')
plt.xlabel('B2c')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import hilbert, periodogram
import os

repertory_dataset = glob('/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_noise/GP80_202501*')[:50]

fs = 5e8 # in Hz

data = np.load(repertory_dataset[0])
# plt.figure(figsize=(10,6))
# fx, Pxx = periodogram(data[0,:,0], fs=fs)
# fy, Pxy = periodogram(data[0,:,1], fs=fs)
# fz, Pxz = periodogram(data[0,:,2], fs=fs)

fx, Pxx = periodogram(data[0,0,:], fs=fs)
fy, Pxy = periodogram(data[0,1,:], fs=fs)
fz, Pxz = periodogram(data[0,2,:], fs=fs)

div = 0

for i in range(len(repertory_dataset)) :
    data = np.load(repertory_dataset[i])
    for j in range(data.shape[0]):
        if i==0 and j==0 :
            continue
        _, Pxx1 = periodogram(data[j,0,:], fs=fs)
        Pxx += Pxx1
        _, Pxy1 = periodogram(data[j,1,:], fs=fs)
        Pxy += Pxy1
        _, Pxz1 = periodogram(data[j,2,:], fs=fs)
        Pxz += Pxz1
        div += 1



plt.semilogy(fx, Pxx/div,
             label='X channel')
plt.semilogy(fy, Pxy/div,
             label='Y channel')
plt.semilogy(fz, Pxz/div,
             label='Z channel')
plt.xlabel('Frequency [Hz]',
           fontsize=14)
plt.ylabel('PSD [ADC^2/Hz]',
           fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(ymin=1e-8)
plt.title(f'PSD for simulated EAS signals',
          wrap=True,
          fontsize=16)
plt.legend(fontsize=12)
os.makedirs('/sps/grand/jlavoisier/output/ML_cuts/plots/', exist_ok=True)
plt.savefig('/sps/grand/jlavoisier/output/ML_cuts/plots/PSD_noise_20_files.png')
plt.show()
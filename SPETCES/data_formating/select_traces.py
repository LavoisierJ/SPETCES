import numpy as np
from glob import glob
import os

from imported_fcts import master_path

"""

This script creates a file with a specified number of traces to construct datasets for ML training and testing.
Traces are selected according to different parameters (noise level, SNR level).
Additionnally, traces are treated so that their mean is centered around 0, and are only 512-sample long, with a randomized pulse position.

"""


repert_simu = sorted(glob(f'{master_path}/datasets/dataset_signal/adc*'))
repert_noise = sorted(glob(f'{master_path}/datasets/dataset_noise/GP80_202507*'))

number_of_traces = 1000

# We want both sims and data to have comparable noise levels
noise_level = 15
std_noise_level = 2

# We want both sims and data to have comparable SNR levels
SNR_level = 4

# We count the number of traces selected for each dataset
count_simu = 0
count_noise = 0

dataset_simu = np.empty((0,512,2))
dataset_noise = np.empty((0,512,2))

# Where to save the datasets
os.makedirs(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/', exist_ok=True)
save_path_simu = f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/raw_dataset/simu_dataset_{number_of_traces}_traces_noise{noise_level}_SNR{SNR_level}.npy'
save_path_noise = f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/raw_dataset/noise_dataset_{number_of_traces}_traces_noise{noise_level}_SNR{SNR_level}.npy'

# To have 0 mean traces
means = np.zeros((2))
traces_0_mean = np.zeros((2, 512))

# ------------------------TRAIN AND VALIDATION SETS -------------------------
# -------------------------------- Sims -------------------

for file in repert_simu:
    data = np.load(file)
    for j in range(data.shape[0]):
        if count_simu >= number_of_traces:
            break
        trace_X = data[j,0,:]
        trace_Y = data[j,1,:]
        # We compute the noise level on the first 1000 samples (20 us)
        sigma_stationnary_X = np.std(trace_X[512:])
        sigma_stationnary_Y = np.std(trace_Y[512:])
        if (sigma_stationnary_X < noise_level + std_noise_level) and (sigma_stationnary_X > noise_level - std_noise_level) and (sigma_stationnary_Y < noise_level + std_noise_level) and (sigma_stationnary_Y > noise_level - std_noise_level):
            # We compute the SNR on the full trace
            SNR_X = np.max(trace_X) / (sigma_stationnary_X)
            SNR_Y = np.max(trace_Y)/ (sigma_stationnary_Y)
            if (SNR_X > SNR_level) or (SNR_Y > SNR_level):
                # We select a random position in the trace such that the signal is fully contained in a 512-sample window
                n = np.random.randint(0, 512)
                while (np.max(trace_X[n:n+512])!= np.max(trace_X)) and (np.max(trace_Y[n:n+512])!= np.max(trace_Y)):
                    n = np.random.randint(0, 512)
                traces_mean = data[j,0:2,n:n+512]
                means[0] = np.mean(traces_mean[0])
                means[1] = np.mean(traces_mean[1])

                traces_0_mean[0] = traces_mean[0] - means[0]
                traces_0_mean[1] = traces_mean[1] - means[1]

                dataset_simu = np.concatenate((dataset_simu, [traces_0_mean.T]), axis=0)
                count_simu += 1
                print(f'Selected {count_simu} traces for the simulated dataset', end='\r')
            else:
                print('SNR insufficient.')
        else:
            print(f'Background noise outside : sigma_X = {sigma_stationnary_X}; sigma_Y = {sigma_stationnary_Y}.')


np.save(save_path_simu, dataset_simu)

# -------------------------------- Noise -------------------

for file in repert_noise:
    if count_noise >= number_of_traces:
        break
    data = np.load(file)
    print(file.split('/')[-1])
    for j in range(data.shape[0]):
        if count_noise >= number_of_traces:
            break
        trace_X = data[j,0,:]
        trace_Y = data[j,1,:]
        # We compute the noise level on the first 1000 samples (20 us)
        sigma_stationnary_X = np.std(trace_X[512:])
        sigma_stationnary_Y = np.std(trace_Y[512:])
        if (sigma_stationnary_X < noise_level + std_noise_level) and (sigma_stationnary_X > noise_level - std_noise_level) and (sigma_stationnary_Y < noise_level + std_noise_level) and (sigma_stationnary_Y > noise_level - std_noise_level):
            # We compute the SNR on the full trace
            # SNR_X = (np.max(trace_X) - np.min(trace_X)) / (2*sigma_stationnary_X)
            # SNR_Y = (np.max(trace_Y) - np.min(trace_Y)) / (2*sigma_stationnary_Y)
            SNR_X = np.max(trace_X)/ (sigma_stationnary_X)
            SNR_Y = np.max(trace_Y) / (sigma_stationnary_Y)
            if (SNR_X > SNR_level) and (SNR_Y > SNR_level):
                # We select a random position in the trace such that the signal is fully contained in a 512-sample window
                n = np.random.randint(0, 512)
                while (np.max(trace_X[n+20:n+512])!= np.max(trace_X)) and (np.max(trace_Y[n+20:n+512])!= np.max(trace_Y)):
                    n = np.random.randint(0, 512)
                traces_mean = data[j,0:2,n:n+512]
                means[0] = np.mean(traces_mean[0])
                means[1] = np.mean(traces_mean[1])

                traces_0_mean[0] = traces_mean[0] - means[0]
                traces_0_mean[1] = traces_mean[1] - means[1]

                dataset_noise = np.concatenate((dataset_noise, [traces_0_mean.T]), axis=0)
                count_noise += 1
                print(f'Selected trace {j} of file.')
                print(f'Selected in total {count_noise} traces for the noise dataset', end='\r')
        #     else:
        #         print('SNR insufficient.')
        # else:
        #     print(f'Background noise outside : sigma_X = {sigma_stationnary_X}; sigma_Y = {sigma_stationnary_Y}.')

np.save(save_path_noise, dataset_noise)


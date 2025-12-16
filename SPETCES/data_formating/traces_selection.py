import numpy as np
from glob import glob
import os

from SPETCES.imported_fcts import master_path

"""

This script creates a file with a specified number of traces to construct datasets for ML training and testing.
Traces are selected according to different parameters (noise level, SNR level).
Additionnally, traces are treated so that their mean is centered around 0, and are only 512-sample long, with a randomized pulse position.

"""


repert_dir = glob(f'{master_path}/gathered_traces/noise/*')[:4]

number_of_traces = 433
suffixe = '433'

type_of_data = repert_dir[0].split('/')[-2]  # 'sims' or 'noise'

# We want both sims and data to have comparable noise levels
noise_level = 15
std_noise_level = 5

# We want both sims and data to have comparable SNR levels
SNR_level = 4

# We count the number of traces selected for each dataset
count_data = 0

dataset_data = np.empty((0,512,2))
dataset_info = np.empty((0,11))

# To have 0 mean traces
means = np.zeros((2))
traces_0_mean = np.zeros((2, 512))

count_events = 0

for i in range(len(repert_dir)):
    data = np.load(repert_dir[i] + '/traces.npy')
    info = np.load(repert_dir[i] + '/info.npy')

    if count_data >= number_of_traces:
        break

    for j in range(np.int32(info[-1,1])): # loop per event
        count_events += 1
        if count_data >= number_of_traces:
            break
        event_info = info[info[:,1]==j]
        trace_events = data[np.int32(event_info[:,0])]

        # We select te traces according to the info parameters recorded, using thresholds set earlier
        for k in range(event_info[:,0].shape[0]): # loop per trace in the event
            trace_info = event_info[k]
            SNR_X = trace_info[5]
            SNR_Y = trace_info[6]
            sigma_stationnary_X = trace_info[7]
            sigma_stationnary_Y = trace_info[8]

            if (sigma_stationnary_X < noise_level + std_noise_level) and (sigma_stationnary_X > noise_level - std_noise_level) and (sigma_stationnary_Y < noise_level + std_noise_level) and (sigma_stationnary_Y > noise_level - std_noise_level):
                if (SNR_X > SNR_level) or (SNR_Y > SNR_level):
                    # We select a random position in the trace such that the signal is fully contained in a 512-sample window
                    n = np.random.randint(0, 512)
                    while (np.max(trace_events[k,0,n:n+512])!= np.max(trace_events[k,0,:])) and (np.max(trace_events[k,1,n:n+512])!= np.max(trace_events[k,1,:])):
                        n = np.random.randint(0, 512)
                    traces_mean = trace_events[k,0:2,n:n+512]
                    means[0] = np.mean(traces_mean[0])
                    means[1] = np.mean(traces_mean[1])

                    traces_0_mean[0] = traces_mean[0] - means[0]
                    traces_0_mean[1] = traces_mean[1] - means[1]

                    dataset_data = np.concatenate((dataset_data, [traces_0_mean.T]), axis=0)

                    dataset_info = np.concatenate((dataset_info, [trace_info]), axis=0)
                    count_data += 1

                    print(f'Selected {count_data} traces for the data dataset', end='\r')

# We change the first column of the info file to have a line number corresponding to the dataset created
for i in range(dataset_info.shape[0]):
    dataset_info[i,0] = i

# Where to save the datasets
os.makedirs(f'{master_path}/datasets/dataset_{suffixe}/raw_dataset/', exist_ok=True)
save_path_data_traces = f'{master_path}/datasets/dataset_{suffixe}/raw_dataset/{type_of_data}_dataset_{dataset_data.shape[0]}_traces_noise{noise_level}_SNR{SNR_level}_traces.npy'
save_path_data_info = f'{master_path}/datasets/dataset_{suffixe}/raw_dataset/{type_of_data}_dataset_{dataset_data.shape[0]}_traces_noise{noise_level}_SNR{SNR_level}_info.npy'


np.save(save_path_data_traces, dataset_data)
np.save(save_path_data_info, dataset_info)
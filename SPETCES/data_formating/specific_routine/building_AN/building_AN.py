import os
import numpy as np
from glob import glob

from SPETCES.imported_fcts import master_path
from SPETCES.data_formating.traces_gathering import pass_T1


"""
This script builds the AN building dataset by selecting traces from preliminary datasets.

"""

# repert_preliminary_datasets = sorted(glob(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/raw_dataset/*'))

repert_NJ = sorted(glob(f'{master_path}/datasets/dataset_building_AN/NJ_sims/adc*'))
repert_noise = sorted(glob(f'{master_path}/datasets/dataset_noise/GP80_202507*'))[:4]

print("Loading noise traces...")
# first we load the noise datasets, and extract the last 512 points from the first two channels
list_noise_traces = np.zeros((0, 512, 2))
for file in repert_noise:
    data = np.load(file)
    traces = data[:, :2, -512:]
    traces = np.swapaxes(traces, 1, 2)
    list_noise_traces = np.append(list_noise_traces, traces, axis=0)
    print(np.shape(list_noise_traces))


# then we extract 512 points of the NJ sims (where the signal should be) and add to them the extracted noise traces
list_NJ_traces = np.zeros((0, 512, 2))
for i, file in enumerate(repert_NJ):
    if np.shape(list_NJ_traces)[0] >= np.shape(list_noise_traces)[0] or np.shape(list_NJ_traces)[0] >= 1000:
        break
    data = np.load(file)
    traces = data[:, :2, :]
    traces_with_noise = np.zeros((np.shape(traces)[0], 512, 2))

    # we select 512 points where the pulse is situated
    for j in range(data.shape[0]):

        print(f'File {i+1}/{len(repert_NJ)} - Trace {j+1}/{data.shape[0]}')
        trace_X = data[j,0,:]
        trace_Y = data[j,1,:]
        n = np.random.randint(0, 512)
        while (np.max(trace_X[n+20:n+512-20])!= np.max(trace_X)) and (np.max(trace_Y[n+20:n+512-20])!= np.max(trace_Y)):
            print('-')
            n = np.random.randint(0, 512)
        traces_with_noise[j,:,0] = trace_X[n:n+512]
        traces_with_noise[j,:,1] = trace_Y[n:n+512]
        

    # we add noise to the traces
    traces_with_noise = traces_with_noise + list_noise_traces[np.shape(list_NJ_traces)[0]:np.shape(list_NJ_traces)[0]+np.shape(traces_with_noise)[0]]
    traces_with_noise = traces_with_noise - np.mean(traces_with_noise, axis=1, keepdims=True)
    list_NJ_traces = np.append(list_NJ_traces, traces_with_noise, axis=0)
    print(np.shape(list_NJ_traces))

print(np.shape(list_NJ_traces[:1000]))
# we save the built dataset
os.makedirs(f'{master_path}/datasets/dataset_building_AN/AN_handmade', exist_ok=True)
save_path_AN_building = f'{master_path}/datasets/dataset_building_AN/AN_handmade/AN_handmade_NJ_sims_with_noise.npy'
np.save(save_path_AN_building, list_NJ_traces[:1000])
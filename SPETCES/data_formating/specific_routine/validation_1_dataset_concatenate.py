import numpy as np
from glob import glob
import grand.dataio.root_trees as rt
import os
import matplotlib.pyplot as plt

datasets_augmented_path = "/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_real_CR2/raw_dataset"

repertory_datasets_traces = sorted(glob(f'{datasets_augmented_path}/*_traces.npy'))
repertory_datasets_info = sorted(glob(f'{datasets_augmented_path}/*_info.npy'))

traces_full = np.zeros((0, 512, 2), dtype=int)
info_full = np.zeros((0, 11))

for i in range(len(repertory_datasets_traces)):
    traces = np.load(repertory_datasets_traces[i])
    info = np.load(repertory_datasets_info[i])
    print(f'File {i} : {traces.shape[0]} traces')

    traces_full = np.append(traces_full, traces, axis=0)
    info_full = np.append(info_full, info, axis=0)

np.save(f'{datasets_augmented_path}/marion_202512_sure_{traces_full.shape[0]}_full_traces.npy', traces_full)
np.save(f'{datasets_augmented_path}/marion_202512_sure_{info_full.shape[0]}_full_info.npy', info_full)

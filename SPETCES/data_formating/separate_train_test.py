"""
This program separates the noise and signal datasets to have two distinct train and test datasets for each.
"""

import numpy as np
from SPETCES.imported_fcts import master_path

# data_noise = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/raw_dataset/noise_dataset_20000_traces_noise15_SNR4.npy')
# data_signal = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/raw_dataset/simu_dataset_20000_traces_noise15_SNR4.npy')

data_signal = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/raw_dataset/AN_handmade_NJ_sims_with_noise.npy')

# data_noise = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/raw_dataset/traces_to_save_mine.npy')
# data_signal = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/raw_dataset/traces_to_save_plane.npy')

test_prop = 0.2

number_data = np.shape(data_signal)[0]
number_test = int(number_data*test_prop)
number_train = number_data - number_test
print(f'Number of test traces: {number_test}, number of train traces: {number_train}')

# Shuffle
liste = np.arange(number_data)
np.random.shuffle(liste)

# Create train and test sets
# train_set_noise = data_noise[liste[:number_train]]
# test_set_noise = data_noise[liste[number_train:]]

train_set_signal = data_signal[liste[:number_train]]
test_set_signal = data_signal[liste[number_train:]]

# Save datasets

np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_train_800_traces_noise15_SNR4.npy', train_set_signal)
np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy', test_set_signal)

# np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/train_test_datasets/noise_dataset_train_16000_traces_noise15_SNR4.npy', train_set_noise)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/train_test_datasets/noise_dataset_test_4000_traces_noise15_SNR4.npy', test_set_noise)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/train_test_datasets/simu_dataset_train_16000_traces_noise15_SNR4.npy', train_set_signal)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/dataset_20k/train_test_datasets/simu_dataset_test_4000_traces_noise15_SNR4.npy', test_set_signal)

# np.save(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/mine_dataset_train_800.npy', train_set_noise)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/mine_dataset_test_200.npy', test_set_noise)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/plane_dataset_train_800.npy', train_set_signal)
# np.save(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/plane_dataset_test_200.npy', test_set_signal)
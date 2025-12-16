"""
This program separates the noise and signal datasets to have two distinct train and test datasets for each.
"""

import numpy as np
from SPETCES.imported_fcts import master_path

dataset_name = 'dataset_real_CR2'

# data_noise = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/noise_dataset_1002_traces_noise15_SNR4_traces.npy')
# info_noise = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/noise_dataset_1002_traces_noise15_SNR4_info.npy')

data_signal = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/marion_202512_sure_1701_full_traces.npy')
info_signal = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/marion_202512_sure_1701_full_info.npy')


test_prop = 0.2

number_data = np.shape(data_signal)[0]
number_test = int(number_data*test_prop)
number_train = number_data - number_test
print(f'Number of test traces: {number_test}, number of train traces: {number_train}')

# Shuffle
liste = np.arange(number_data)
np.random.shuffle(liste)

# creer une façon de séparer les evenements en entier de façon à avoir tout de même des events complets dans le train et dans le test

# Create train and test sets
# train_set_noise = data_noise[liste[:number_train]]
# test_set_noise = data_noise[liste[number_train:]]

# train_info_noise = info_noise[liste[:number_train]]
# test_info_noise = info_noise[liste[number_train:]]


train_set_signal = data_signal[liste[:number_train]]
test_set_signal = data_signal[liste[number_train:]]

test_info_signal = info_signal[liste[number_train:]]
train_info_signal = info_signal[liste[:number_train]]

# Save datasets

# Train
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/mine_dataset_train_800_traces_noise25_SNR4_traces.npy', train_set_noise)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/mine_dataset_train_800_traces_noise25_SNR4_info.npy', train_info_noise)

np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_real_dataset_train_{number_train}_traces_noise25_SNR4_traces.npy', train_set_signal)
np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_real_dataset_train_{number_train}_traces_noise25_SNR4_info.npy', train_info_signal)

# Test
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/mine_dataset_test_200_traces_noise25_SNR4_traces.npy', test_set_noise)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/mine_dataset_test_200_traces_noise25_SNR4_info.npy', test_info_noise)

np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_real_dataset_test_{number_test}_traces_noise25_SNR4_traces.npy', test_set_signal)
np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_real_dataset_test_{number_test}_traces_noise25_SNR4_info.npy', test_info_signal)

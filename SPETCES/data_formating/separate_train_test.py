"""
This program separates the noise and signal datasets to have two distinct train and test datasets for each.
"""

import numpy as np
from SPETCES.imported_fcts import master_path

dataset_name = 'dataset_real_CR4_384_1024trace'

# data_noise = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/noise_dataset_1002_traces_noise15_SNR4_traces.npy')
# info_noise = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/noise_dataset_1002_traces_noise15_SNR4_info.npy')

data_signal = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/marion_202512_sure_1812_full_traces.npy')
info_signal = np.load(f'{master_path}/datasets/{dataset_name}/raw_dataset/marion_202512_sure_1812_full_info.npy')


test_prop = 0.2

number_data_signal = np.shape(data_signal)[0]
number_test_signal = int(number_data_signal*test_prop)
number_train_signal = number_data_signal - number_test_signal
print(f'Number of test traces: {number_test_signal}, number of train traces: {number_train_signal}')

# number_data_noise = np.shape(data_noise)[0]
# number_test_noise = int(number_data_noise*test_prop)
# number_train_noise = number_data_noise - number_test_noise

# Shuffle
liste_signal = np.arange(number_data_signal)
np.random.shuffle(liste_signal)

# liste_noise = np.arange(number_data_noise)
# np.random.shuffle(liste_noise)

# creer une façon de séparer les evenements en entier de façon à avoir tout de même des events complets dans le train et dans le test

# Create train and test sets
# train_set_noise = data_noise[liste_noise[:number_train_noise]]
# test_set_noise = data_noise[liste_noise[number_train_noise:]]

# train_info_noise = info_noise[liste_noise[:number_train_noise]]
# test_info_noise = info_noise[liste_noise[number_train_noise:]]



train_set_signal = data_signal[liste_signal[:number_train_signal]]
test_set_signal = data_signal[liste_signal[number_train_signal:]]

test_info_signal = info_signal[liste_signal[number_train_signal:]]
train_info_signal = info_signal[liste_signal[:number_train_signal]]

# Save datasets

# Train
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/train_noise_dataset_{number_train_noise}_traces_noise15_SNR4_traces.npy', train_set_noise)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/train_noise_dataset_{number_train_noise}_traces_noise15_SNR4_info.npy', train_info_noise)

# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/train_sims_AN_dataset_{number_train_signal}_traces_noise15_SNR4_traces.npy', train_set_signal)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/train_sims_AN_dataset_{number_train_signal}_traces_noise15_SNR4_info.npy', train_info_signal)


np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_train_dataset_{number_train_signal}_traces_noise15_SNR4_traces.npy', train_set_signal)
np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_train_dataset_{number_train_signal}_traces_noise15_SNR4_info.npy', train_info_signal)

# Test
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/test_noise_dataset_{number_test_noise}_traces_noise15_SNR4_traces.npy', test_set_noise)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/test_noise_dataset_{number_test_noise}_traces_noise15_SNR4_info.npy', test_info_noise)

# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/test_sims_AN_dataset_{number_test_signal}_traces_noise15_SNR4_traces.npy', test_set_signal)
# np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/test_sims_AN_dataset_{number_test_signal}_traces_noise15_SNR4_info.npy', test_info_signal)


np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_test_dataset_{number_test_signal}_traces_noise15_SNR4_traces.npy', test_set_signal)
np.save(f'{master_path}/datasets/{dataset_name}/train_test_datasets/CR_test_dataset_{number_test_signal}_traces_noise15_SNR4_info.npy', test_info_signal)

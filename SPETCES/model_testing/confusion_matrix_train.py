"""

Testing the trained models with respect to the SNR of the input data.

"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import measure_SNR, plot_predict_wrt_SNR, plot_predict_wrt_SNR_CRC, master_path, path_weights_1D, path_weights_2D

# -------- Importing the models ---------
model_1D = model_1D_def()
model_2D = model_2D_def()
model_1D.load_weights(path_weights_1D + 'model_1D_RUNTEST_2_adam.weights.h5')
model_2D.load_weights(path_weights_2D + 'model_2D_RUNTEST_2_adam.weights.h5')

# ---------- Importing test data -----------

# data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
# data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

data_noise_test = np.load(f'{master_path}/datasets/RUNTEST_2/test_dataset_RUNTEST_2_3566_traces.npy')
info_noise_test = np.load(f'{master_path}/datasets/RUNTEST_2/test_dataset_RUNTEST_2_3566_info.npy')
data_signal_test = np.empty((0,512,2))  # No signal data for this test

for i in range(info_noise_test.shape[0]) :
    info_noise_test[i,0] = i  # Re-indexing the trace IDs

number_data_noise_test = np.shape(data_noise_test)[0]
number_data_signal_test = np.shape(data_signal_test)[0]

true_noise_test = np.zeros(number_data_noise_test)
true_signal_test = np.zeros(number_data_signal_test) + 1

data_test = np.append(data_noise_test, 
                      data_signal_test, 
                      axis=0)
true_test = np.append(true_noise_test,
                      true_signal_test,
                      axis=0)

# # Shuffle
# liste_test = np.arange(number_data_signal_test + number_data_noise_test) 
# np.random.shuffle(liste_test)
# data_test = data_test[liste_test]
# true_test = true_test[liste_test]


# ---------- Compile models -----------

model_1D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_2D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ---------- Predictions -----------

prediction_1D = model_1D.predict(data_test,
                                #  true_test, 
                                #  verbose=0
                                 )
prediction_2D = model_2D.predict(data_test,  
                                #  true_test, 
                                #  verbose=0
                                 )

print(prediction_1D)
# SNR_test = measure_SNR(data_test)
SNR_test = np.max(info_noise_test[:,5:7], axis=1)  # SNR is already measured and stored in the info file
print("SNR measured.")

plot_predict_wrt_SNR(SNR_test,
                     prediction_1D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/RUNTEST_2/model1D_prediction_wrt_SNR',
                     display=False,
                     title='Model 1D Predictions vs SNR',
                     predict_mean=True)

plot_predict_wrt_SNR(SNR_test,
                     prediction_2D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/RUNTEST_2/model2D_prediction_wrt_SNR',
                     display=False,
                     title='Model 2D Predictions vs SNR',
                     predict_mean=True)
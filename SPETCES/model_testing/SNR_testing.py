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
model_1D.load_weights(path_weights_1D + 'model_1D_ANhm_1k_adam.weights.h5')
model_2D.load_weights(path_weights_2D + 'model_2D_ANhm_1k_adam.weights.h5')

# ---------- Importing test data -----------

data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

number_data_test = np.shape(data_noise_test)[0]


true_noise_test = np.zeros(number_data_test)
true_signal_test = np.zeros(number_data_test) + 1

data_test = np.append(data_noise_test, 
                      data_signal_test, 
                      axis=0)
true_test = np.append(true_noise_test,
                      true_signal_test,
                      axis=0)

# Shuffle
liste_test = np.arange(number_data_test*2) # as both datasets are of equal lengths
np.random.shuffle(liste_test)
data_test = data_test[liste_test]
true_test = true_test[liste_test]


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
SNR_test = measure_SNR(data_test)
print("SNR measured.")

plot_predict_wrt_SNR(SNR_test,
                     prediction_1D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/ANhm_1k_dataset/model1D_prediction_wrt_SNR',
                     display=False,
                     title='Model 1D Predictions vs SNR',
                     predict_mean=True)

plot_predict_wrt_SNR(SNR_test,
                     prediction_2D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/ANhm_1k_dataset/model2D_prediction_wrt_SNR',
                     display=False,
                     title='Model 2D Predictions vs SNR',
                     predict_mean=True)


# # ---------- Add predictions over ICRC2025 cosmics -----------
# repert_predict = sorted(glob(f'{master_path}/datasets/dataset_verif/ICRC2025/*'))

# pred_ICRC2025_1D = np.array([])
# pred_ICRC2025_2D = np.array([])

# for i in range(len(repert_predict)) :
#     data_test = np.load(repert_predict[i])

#     name = repert_predict[i].split('/')[-1]

#     pred_1D = model_1D.predict(data_test)
#     pred_2D = model_2D.predict(data_test)

#     pred_ICRC2025_1D = np.append(pred_ICRC2025_1D, pred_1D)
#     pred_ICRC2025_2D = np.append(pred_ICRC2025_2D, pred_2D)

#     SNR_test = np.append(SNR_test, measure_SNR(data_test))


# true_ICRC2025 = np.zeros(len(pred_ICRC2025_1D)) + 2 # label 2 for ICRC cosmics


# true_test = np.append(true_test,
#                       true_ICRC2025,
#                       axis=0)

# pred_ICRC2025_1D = pred_ICRC2025_1D.reshape(-1,1)
# pred_ICRC2025_2D = pred_ICRC2025_2D.reshape(-1,1)

# prediction_1D =  np.append(prediction_1D, pred_ICRC2025_1D, axis=0)
# prediction_2D =  np.append(prediction_2D, pred_ICRC2025_2D, axis=0)

# plot_predict_wrt_SNR_CRC(SNR_test,
#                          prediction_1D,
#                          true_labels=true_test,
#                          save_path=f'{master_path}/plots/model1D_prediction_wrt_SNR_ICRC2025',
#                          display=False,
#                          title='Model 1D Predictions vs SNR (including ICRC2025 cosmics)')

# plot_predict_wrt_SNR_CRC(SNR_test,
#                          prediction_2D,
#                          true_labels=true_test,
#                          save_path=f'{master_path}/plots/model2D_prediction_wrt_SNR_ICRC2025',
#                          display=False,
#                          title='Model 2D Predictions vs SNR (including ICRC2025 cosmics)')
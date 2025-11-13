"""

Testing the trained models used for average model with respect to the SNR of the input data.

"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import measure_SNR, plot_predict_averaged_wrt_SNR, master_path, path_weights_average_1D, path_weights_average_2D, plot_predict_averaged_wrt_SNR_CRC


# -------- Importing the models ---------

model_1D_list = np.array([model_1D_def() for _ in range(10)])
model_2D_list = np.array([model_2D_def() for _ in range(10)])

for i, model in enumerate(model_1D_list):
    model.load_weights(path_weights_average_1D + f'model_1D_adam_{i}.weights.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

for i, model in enumerate(model_2D_list):
    model.load_weights(path_weights_average_2D + f'model_2D_adam_{i}.weights.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# ---------- Importing test data -----------

data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

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



# ---------- Predictions -----------
prediction_1D_list = np.empty((data_test.shape[0], len(model_1D_list)))
prediction_2D_list = np.empty((data_test.shape[0], len(model_2D_list)))

for i, model in enumerate(model_1D_list):
    prediction_1D_list[:, i] = np.reshape(model.predict(data_test, 
                                            #  true_test, 
                                            #  verbose=0
                                             ), (-1,))

for i, model in enumerate(model_2D_list):
    prediction_2D_list[:, i] = np.reshape(model.predict(data_test,  
                                            #  true_test, 
                                            #  verbose=0
                                             ), (-1,))

SNR_test = measure_SNR(data_test)

# plot_predict_averaged_wrt_SNR(SNR_test,
#                      prediction_1D_list,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/averaged/model1D_prediction_wrt_SNR',
#                      display=False,
#                      title='Model 1D Mean Predictions vs SNR',
#                      predict_mean=True)

# plot_predict_averaged_wrt_SNR(SNR_test,
#                      prediction_2D_list,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/averaged/model2D_prediction_wrt_SNR',
#                      display=False,
#                      title='Model 2D Mean Predictions vs SNR',
#                      predict_mean=True)


# ---------- Add predictions over ICRC2025 cosmics -----------

origin_CRC = 'pengxiong_202510'

repert_predict = sorted(glob(f'{master_path}/datasets/dataset_verif/{origin_CRC}/*'))

data_CRC = np.empty((0, 512, 2))
for i in range(len(repert_predict)) :
    data_CRC_i = np.load(repert_predict[i])
    data_CRC = np.append(data_CRC, data_CRC_i, axis=0)


pred_CRC_1D = np.empty((data_CRC.shape[0],len(model_1D_list)))
pred_CRC_2D = np.empty((data_CRC.shape[0],len(model_2D_list)))

for i, model in enumerate(model_1D_list):
    pred_CRC_1D[:, i] = np.reshape(model.predict(data_CRC, 
                                            #  true_test, 
                                            #  verbose=0
                                             ), (-1,))

for i, model in enumerate(model_2D_list):
    pred_CRC_2D[:, i] = np.reshape(model.predict(data_CRC,  
                                            #  true_test, 
                                            #  verbose=0
                                             ), (-1,))

SNR_test = np.append(SNR_test, measure_SNR(data_CRC))

true_CRC = np.zeros(np.shape(pred_CRC_1D)[0]) + 2 # label 2 for ICRC cosmics


true_test = np.append(true_test,
                      true_CRC,
                      axis=0)

prediction_1D_list =  np.append(prediction_1D_list, pred_CRC_1D, axis=0)
prediction_2D_list =  np.append(prediction_2D_list, pred_CRC_2D, axis=0)

plot_predict_averaged_wrt_SNR_CRC(SNR_test,
                                prediction_1D_list,
                                true_labels=true_test,
                                save_path=f'{master_path}/plots/averaged/model1D_prediction_wrt_SNR_{origin_CRC}',
                                display=False,
                                title=f'Model 1D Mean Predictions vs SNR (including {origin_CRC} cosmics)')

plot_predict_averaged_wrt_SNR_CRC(SNR_test,
                                prediction_2D_list,
                                true_labels=true_test,
                                save_path=f'{master_path}/plots/averaged/model2D_prediction_wrt_SNR_{origin_CRC}',
                                display=False,
                                title=f'Model 2D Mean Predictions vs SNR (including {origin_CRC} cosmics)')
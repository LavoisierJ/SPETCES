"""
Produce accuracy plots of the model.
"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def, build_cnn1d_resnet, build_cnn2d_resnet
from SPETCES.imported_fcts import measure_SNR, plot_loghist, plot_predict_wrt_SNR, plot_predict_wrt_SNR_CRC, four_notch_filters, confusion_matrix, accuracy_precison_recall, binary_crossentropy, master_path, path_weights_1D, path_weights_2D, fs


dataset_name = 'heavymodel_ANhm_6_384_1024trace'
name_of_dataset = 'heavymodel_realCR_ANhm_6_384_1024trace'

training_acc_1D = np.load(f'./training_acc_1D_{name_of_dataset}_adam.npy')
#training_acc_1D = np.load(f'{master_path}/datasets/{dataset_name}/compiling_results/training_acc_1D_{dataset_name}_adam.npy')
training_acc_2D = np.load(f'{master_path}/datasets/{dataset_name}/compiling_results/training_acc_2D_{dataset_name}_adam.npy')

accuracy_1D = training_acc_1D[0]
val_accuracy_1D = training_acc_1D[1]
loss_1D = training_acc_1D[2]
val_loss_1D = training_acc_1D[3]

accuracy_2D = training_acc_2D[0]
val_accuracy_2D = training_acc_2D[1]
loss_2D = training_acc_2D[2]
val_loss_2D = training_acc_2D[3]

plt.figure(figsize=(8,6))
plt.plot(loss_1D, 
         label='1D model train')
plt.plot(val_loss_1D, 
         label='1D model val')

plt.xlabel('Epochs', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Loss', fontsize=14)
plt.yticks(fontsize=12)
plt.title('1 - Accuracy during training', fontsize=16)
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig(f'losses_1D_{name_of_dataset}_adam.png')
plt.close()
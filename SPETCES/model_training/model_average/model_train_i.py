"""

This code trains two neural network models (1D and 2D), defined in model.py, to classify traces as signal or noise using TensorFlow and Keras.

"""


import tensorflow as tf 
import numpy as np 
import sys

from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import master_path, path_weights_average_1D, path_weights_average_2D

# -------- Importing the models ---------
i = sys.argv[1]

model_1D = model_1D_def()
model_2D = model_2D_def()

epochs = 2000
batch_size = 128

model_1D.compile(loss="binary_crossentropy", 
                 optimizer="adam", 
                 metrics=["accuracy"])
model_2D.compile(loss="binary_crossentropy", 
                 optimizer="adam", 
                 metrics=["accuracy"])




# ---------- Training data -----------

data_noise_train = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_train_800_traces_noise15_SNR4.npy')
data_signal_train = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/simu_dataset_train_800_traces_noise15_SNR4.npy')

number_data_train = np.shape(data_noise_train)[0]


true_noise_train = np.zeros(number_data_train)
true_signal_train = np.zeros(number_data_train) + 1

data_train = np.append(data_noise_train, 
                       data_signal_train, 
                       axis=0)

true_train = np.append(true_noise_train,
                       true_signal_train,
                       axis=0)

# Shuffle
liste_train = np.arange(number_data_train*2) # as both datasets are of equal lengths
np.random.shuffle(liste_train)

data_train = data_train[liste_train]
true_train = true_train[liste_train]



history_1D=model_1D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.1
                        # validation_data=(data_test, true_test)
                        )

history_2D=model_2D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.1
                        # validation_data=(data_test, true_test)
                        )

model_1D.save_weights(path_weights_average_1D + f'model_1D_adam_{i}.weights.h5')
model_2D.save_weights(path_weights_average_2D + f'model_2D_adam_{i}.weights.h5')



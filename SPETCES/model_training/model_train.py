"""

This code trains two neural network models (1D and 2D), defined in model.py, to classify traces as signal or noise using TensorFlow and Keras.

"""


import tensorflow as tf 
import numpy as np 
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def, LRreducer, printlearningrate, LearningRateLogger
from SPETCES.imported_fcts import plot_loss, plot_accuracy, plot_learning_rate, master_path, path_weights_1D, path_weights_2D

tf.config.run_functions_eagerly(True)

# -------- Importing the models ---------
model_1D = model_1D_def()
model_2D = model_2D_def()

epochs = 2000
batch_size = 4
learning_rate_initial = 0.001

name_of_dataset = 'dataset_real_CR2'


# optimizer_1d = tf.keras.optimizers.Adam(learning_rate=LRreducer(initial_learning_rate=0.001,
#                                                              patience=15))
# # Create a callback to update the learning rate
# lr_scheduler_1d = tf.keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: lr_schedule_1d.on_epoch_end(epoch, logs)
# )
# lr_schedule_1d = LRreducer(initial_learning_rate=0.001,
#                            patience=15,
#                            divide=1.5)

lr_schedule_1d = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.66,
                patience=15,
                verbose=0,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=1e-7,
            )
# printlr_1d = printlearningrate()


# optimizer_2d = tf.keras.optimizers.Adam(learning_rate=LRreducer(initial_learning_rate=0.001,
#                                                              patience=15))
# # Create a callback to update the learning rate
# lr_scheduler_2d = tf.keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: lr_schedule_2d.on_epoch_end(epoch, logs)
# )
# lr_schedule_2d = LRreducer(initial_learning_rate=0.001,
#                            patience=15,
#                            divide=1.5)

lr_schedule_2d = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.66,
                patience=15,
                verbose=0,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=1e-7,
            )

# printlr_2d = printlearningrate()


model_1D.compile(loss="binary_crossentropy", 
                #  optimizer="adam", 
                 optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_initial),
                 metrics=["accuracy"])

model_2D.compile(loss="binary_crossentropy", 
                #  optimizer="adam", 
                 optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_initial),
                 metrics=["accuracy"])


# ---------- Training data -----------



# Training data 
data_noise_train = np.load(f'{master_path}/datasets/{name_of_dataset}/train_test_datasets/noise_dataset_dataset_real_CR2_2506_traces.npy')[:1400]

data_signal_train = np.load(f'{master_path}/datasets/{name_of_dataset}/train_test_datasets/CR_real_dataset_train_1361_traces_noise25_SNR4_traces.npy')

# data_signal_train1 = np.load(f'{master_path}/datasets/{name_of_dataset}/marion_202512_sure_567_traces.npy')
# data_signal_train2 = np.load(f'{master_path}/datasets/{name_of_dataset}/AN_handmade_dataset_RUNTEST_6_realCR_ANhm_noise_sigma_10_20_1525_traces.npy')[:433]

# data_signal_train = np.append(data_signal_train1,
#                               data_signal_train2,
#                               axis=0)

print('------------------')
print(np.shape(data_noise_train))
print(np.shape(data_signal_train))
print('------------------')
number_data_noise_train = np.shape(data_noise_train)[0]
number_data_signal_train = np.shape(data_signal_train)[0]

true_noise_train = np.zeros(number_data_noise_train)
true_signal_train = np.zeros(number_data_signal_train) + 1

data_train = np.append(data_noise_train, 
                       data_signal_train, 
                       axis=0)

true_train = np.append(true_noise_train,
                       true_signal_train,
                       axis=0)

# Shuffle
liste_train = np.arange(data_train.shape[0])
np.random.shuffle(liste_train)

data_train = data_train[liste_train]
true_train = true_train[liste_train]

# # ---------------------------
# # Testing data
# data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
# data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

# number_data_noise_test = np.shape(data_noise_test)[0]
# number_data_signal_test = np.shape(data_signal_test)[0]

# true_noise_test = np.zeros(number_data_noise_test)
# true_signal_test = np.zeros(number_data_signal_test) + 1

# data_test = np.append(data_noise_test, 
#                       data_signal_test, 
#                       axis=0)
# true_test = np.append(true_noise_test,
#                       true_signal_test,
#                       axis=0)

# # Shuffle
# liste_test = np.arange(data_test.shape[0])
# np.random.shuffle(liste_test)

# data_test = data_test[liste_test]
# true_test = true_test[liste_test]

# --------- Training the models ---------

os.makedirs(f'{master_path}/plots/{name_of_dataset}', exist_ok=True)

history_1D=model_1D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.1,
                        callbacks=[LearningRateLogger(),
                                   lr_schedule_1d],
                        # verbose=0
                        # validation_data=(data_test, true_test)
                        )
model_1D.save_weights(path_weights_1D + f'model_1D_{name_of_dataset}_adamW.weights.h5')

plot_loss(history_1D,
          save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adamW_loss',
          display=False)
plot_accuracy(history_1D,
              save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adamW_accuracy',
              display=False)
plot_learning_rate(history_1D,
                   save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adamW_learning_rate',
                   display=False)


print('\n\n\n\n\n----------------------------------')
print('Starting training of 2D model')
print('----------------------------------\n\n\n\n\n')
history_2D=model_2D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.1,
                        callbacks=[LearningRateLogger(), 
                                   lr_schedule_2d]
                        # validation_data=(data_test, true_test)
                        )
model_2D.save_weights(path_weights_2D + f'model_2D_{name_of_dataset}_adamW.weights.h5')


plot_loss(history_2D,
          save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adamW_loss',
          display=False)
plot_accuracy(history_2D,
              save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adamW_accuracy',
              display=False)
plot_learning_rate(history_2D,
                   save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adamW_learning_rate',
                   display=False)

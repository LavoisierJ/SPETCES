"""

This code trains two neural network models (1D and 2D), defined in model.py, to classify traces as signal or noise using TensorFlow and Keras.

"""


import tensorflow as tf 
import numpy as np 
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def, build_cnn1d_resnet, build_cnn2d_resnet, LRreducer, printlearningrate, LearningRateLogger, MyLRSchedule
from SPETCES.imported_fcts import plot_loss, plot_accuracy, plot_learning_rate, master_path, path_weights_1D, path_weights_2D

#tf.config.run_functions_eagerly(True)

# -------- Importing the models ---------
# model_1D = model_1D_def(trace_shape=384)
# model_2D = model_2D_def(trace_shape=384)

model_1D = build_cnn1d_resnet(
    input_shape=(200, 2),
    filters=16,
                            #   kernel_size=3
    )
model_2D = build_cnn2d_resnet(
    input_shape=(384, 2, 1), 
    filters=4, 
                            #   kernel_size=3
    )

nb_epochs = 1
batch_size = 16
learning_rate_initial = 0.003
my_lrs = MyLRSchedule(base_lr=0.00005, max_lr=0.001, step_size=500)


name_of_dataset = 'heavymodel_realCR_6_384_1024trace'
name_of_dataset = 'heavymodel_realCR_ANhm_6_384_1024trace'

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
                monitor='loss',
                factor=0.66,
                patience=30,
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
                monitor='loss',
                factor=0.66,
                patience=30,
                verbose=0,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=1e-7,
            )

# printlr_2d = printlearningrate()


model_1D.compile(loss="binary_crossentropy", 
                 optimizer=tf.keras.optimizers.Adam(learning_rate=my_lrs),
                #  optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_initial,
                #  weight_decay=0.004),
                metrics=["accuracy"])


#model_1D.load_weights(f'./model_1D_{name_of_dataset}_adam_save.weights.h5')

model_2D.compile(loss="binary_crossentropy", 
                 optimizer="adam", 
                #  optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_initial,
                #  weight_decay=0.004),
                 metrics=["accuracy"])


# ---------- Training data -----------

# Training data 
data_noise_train = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/noise_dataset_train_2497_traces_noise15_SNR4_traces.npy')[:1800]
data_noise_validation = data_noise_train[:int(np.shape(data_noise_train)[0]*0.1)] #validation is 10 % of the training set
data_noise_train = data_noise_train[int(np.shape(data_noise_train)[0]*0.1):] #training is 90 % of the training set

data_signal_train = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/CR_train_dataset_1304_traces_noise15_SNR4_traces.npy')
data_signal_validation = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/CR_validation_dataset_144_traces_noise15_SNR4_traces.npy')

# data_signal_train = np.load(f'{master_path}/datasets/{name_of_dataset}/train_test_datasets/ANhm_dataset_train_2143_traces_noise15_SNR4_traces.npy')
# data_signal_validation = data_signal_train[:int(np.shape(data_signal_train)[0]*0.1)] #validation is 10 % of the training set
# data_signal_train = data_signal_train[int(np.shape(data_signal_train)[0]*0.1):] #training is 90 % of the training set


data_signal_train1 = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/CR_train_dataset_1304_traces_noise15_SNR4_traces.npy')
data_signal_train2 = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/ANhm_dataset_train_2143_traces_noise15_SNR4_traces.npy')


data_signal_validation_1 = np.load(f'/sps/grand/blevy/data/{name_of_dataset}/train_test_datasets/CR_validation_dataset_144_traces_noise15_SNR4_traces.npy')
data_signal_validation_2 = data_signal_train2[:int(np.shape(data_signal_train2)[0]*0.1)] #validation is 10 % of the training set

data_signal_train2 = data_signal_train2[int(np.shape(data_signal_train2)[0]*0.1):] #training is 90 % of the training set

data_signal_train = np.append(data_signal_train1,
                              data_signal_train2,
                              axis=0)

data_signal_validation = np.append(data_signal_validation_1,
                                  data_signal_validation_2,
                                  axis=0)


# ---------- Train treatment ---------------------
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

# ------------------------- Validation treatment -------------------------
number_data_noise_validation = np.shape(data_noise_validation)[0]
number_data_signal_validation = np.shape(data_signal_validation)[0]

true_noise_validation = np.zeros(number_data_noise_validation)
true_signal_validation = np.zeros(number_data_signal_validation) + 1

data_validation = np.append(data_noise_validation, 
                      data_signal_validation, 
                      axis=0)
true_validation = np.append(true_noise_validation,
                      true_signal_validation,
                      axis=0)

# Shuffle
liste_validation = np.arange(data_validation.shape[0])
np.random.shuffle(liste_validation)

data_validation = data_validation[liste_validation]
true_validation = true_validation[liste_validation]

# --------- Training the models ---------

os.makedirs(f'./plots/{name_of_dataset}', exist_ok=True)
os.makedirs(f'./{name_of_dataset}/compiling_results', exist_ok=True)

data_train = data_train[:, 0:200, :]*1.0
data_validation = data_validation[:, 0:200, :]*1.0

data_train_mean = data_train.mean()
data_train_std  = data_train.std()

data_train -= data_train_mean
data_train /= data_train_std


data_validation -= data_train_mean
data_validation /= data_train_std

n_sample = np.shape(data_train)[0]


# history_1D1 = model_1D.fit(
#     data_train[0:25],
#     true_train[0:25],
#     batch_size=1,
#     epochs=10,
#     steps_per_epoch=250,
#     # validation_split=0.1,
#     callbacks=[LearningRateLogger()],
#     #            lr_schedule_1d],
#     # verbose=0
#     validation_data=(data_validation[0:10], true_validation[0:10])
# )



history_1D2 = model_1D.fit(
    data_train,
    true_train,
    batch_size=n_sample,
    epochs=5000,
    #steps_per_epoch=500,
    # validation_split=0.1,
    callbacks=[LearningRateLogger()],
    #            lr_schedule_1d],
    # verbose=0
    validation_data=(data_validation, true_validation)
)


model_1D.save_weights(f'./model_1D_{name_of_dataset}_adam.weights.h5')

accuracy_1D = history_1D2.history['accuracy']
val_accuracy_1D = history_1D2.history['val_accuracy']
loss_1D = history_1D2.history['loss']
val_loss_1D = history_1D2.history['val_loss']
lr_1D = history_1D2.history['learning_rate']
training_acc_1D = np.array([accuracy_1D, val_accuracy_1D, loss_1D, val_loss_1D, lr_1D])
np.save(f'./training_acc_1D_{name_of_dataset}_adam.npy', training_acc_1D)




if False:


    plot_loss(history_1D,
            save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adam_loss.pdf',
            display=False)



    # plot_loss(history_1D,
    #           save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adam_loss',
    #           display=False)

    # plot_accuracy(history_1D,
    #               save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adam_accuracy',
    #               display=False)
    plot_accuracy(history_1D,
                save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adam_accuracy.pdf',
                display=False)
    # plot_learning_rate(history_1D,
    #                    save_path=f'{master_path}/plots/{name_of_dataset}/model1D_{name_of_dataset}_adam_learning_rate',
    #                    display=False)


   

    print('\n\n\n\n\n----------------------------------')
    print('Starting training of 2D model')
    print('----------------------------------\n\n\n\n\n')
    history_2D=model_2D.fit(data_train, 
                            true_train, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.1,
                            # callbacks=[LearningRateLogger(), 
                            #            lr_schedule_2d]
                            # validation_data=(data_test, true_test)
                            )
    model_2D.save_weights(path_weights_2D + f'model_2D_{name_of_dataset}_adam.weights.h5')

    accuracy_2D = history_2D.history['accuracy']
    val_accuracy_2D = history_2D.history['val_accuracy']
    loss_2D = history_2D.history['loss']
    val_loss_2D = history_2D.history['val_loss']

    training_acc_2D = np.array([accuracy_2D, val_accuracy_2D, loss_2D, val_loss_2D])

    plot_loss(history_2D,
            save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adam_loss.pdf',
            display=False)
    plot_accuracy(history_2D,
                save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adam_accuracy.pdf',
                display=False)
    # plot_learning_rate(history_2D,
    #                    save_path=f'{master_path}/plots/{name_of_dataset}/model2D_{name_of_dataset}_adamW_learning_rate',
    #                    display=False)


    np.save(f'{master_path}/datasets/{name_of_dataset}/compiling_results/training_acc_2D_{name_of_dataset}_adam.npy', training_acc_2D)
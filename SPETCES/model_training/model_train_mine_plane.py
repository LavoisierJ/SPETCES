import tensorflow as tf 
import numpy as np 
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import plot_loss, plot_accuracy, master_path, path_weights_mine_plain_1D, path_weights_mine_plain_2D

# -------- Importing the models ---------
model_1D = model_1D_def()
model_2D = model_2D_def()

epochs = 100
batch_size = 128

model_1D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_2D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# -------- Creating checkpoints (saving weights) -----------

# checkpoint_1D_path = "/Users/jolan/Documents/GRAND_Work/Pipeline_ML/machine_learning/sans_les_mains/checkpoint_models/model_1D.ckpt"
# checkpoint_1dDdir = os.path.dirname(checkpoint_1D_path)

# # Create a callback that saves the model's weights
# cp_1D_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_1D_path,
#                                                     save_weights_only=True,
#                                                     verbose=1)

# checkpoint_2D_path = "/Users/jolan/Documents/GRAND_Work/Pipeline_ML/machine_learning/sans_les_mains/checkpoint_models/model_2D.ckpt"
# checkpoint_2D_dir = os.path.dirname(checkpoint_2D_path)

# # Create a callback that saves the model's weights
# cp_2D_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_2D_path,
#                                                     save_weights_only=True,
#                                                     verbose=1)

# ---------- Training data -----------

data_mine_train = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/mine_dataset_train_800.npy')
data_plane_train = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/plane_dataset_train_800.npy')

data_mine_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/mine_dataset_test_200.npy')
data_plane_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_mine_plane/train_test_datasets/plane_dataset_test_200.npy')


number_data_train = np.shape(data_mine_train)[0]
number_data_test = np.shape(data_mine_test)[0]


true_mine_train = np.zeros(number_data_train)
true_plane_train = np.zeros(number_data_train) + 1

data_train = np.append(data_mine_train, 
                       data_plane_train, 
                       axis=0)

true_train = np.append(true_mine_train,
                       true_plane_train,
                       axis=0)


true_mine_test = np.zeros(number_data_test)
true_plane_test = np.zeros(number_data_test) + 1

data_test = np.append(data_mine_test, 
                      data_plane_test, 
                      axis=0)
true_test = np.append(true_mine_test,
                      true_plane_test,
                      axis=0)

# Shuffle
liste_train = np.arange(number_data_train*2) # as both datasets are of equal lengths
np.random.shuffle(liste_train)

data_train = data_train[liste_train]
true_train = true_train[liste_train]

liste_test = np.arange(number_data_test*2) # as both datasets are of equal lengths
np.random.shuffle(liste_test)

data_test = data_test[liste_test]
true_test = true_test[liste_test]


history_1D=model_1D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(data_test, true_test)
                        )

history_2D=model_2D.fit(data_train, 
                        true_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(data_test, true_test)
                        )

model_1D.save_weights(path_weights_mine_plain_1D)
model_2D.save_weights(path_weights_mine_plain_2D)

os.makedirs(f'{master_path}/plots/mine_plane/', exist_ok=True)
plot_loss(history_1D,
          save_path=f'{master_path}/plots/mine_plane/model1D_mine_plane_loss',
          display=False)
plot_accuracy(history_1D,
              save_path=f'{master_path}/plots/mine_plane/model1D_mine_plane_accuracy',
              display=False)

plot_loss(history_2D,
          save_path=f'{master_path}/plots/mine_plane/model2D_mine_plane_loss',
          display=False)
plot_accuracy(history_2D,
              save_path=f'{master_path}/plots/mine_plane/model2D_mine_plane_accuracy',
              display=False)


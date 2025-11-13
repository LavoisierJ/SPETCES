import numpy as np 
import tensorflow as tf
# import tensorflow_addons as tfa

INIT_LR = 1e-4
MAX_LR = 1e-2
regul=0

def model_1D_def():
    """
    1D Model -- taken from Sandra le Coz's work
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(512, 2)),
        tf.keras.layers.Conv1D(32, 
                               kernel_size=(11,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(32, 
                               kernel_size=(11,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(32, 
                               kernel_size=(11,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_2D_def():
    """
    2D Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(512, 2, 1)),
        tf.keras.layers.Conv2D(32, 
                               kernel_size=(11,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, 
                               kernel_size=(11,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, 
                               kernel_size=(11,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


class LRreducer(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate reducer that halves the learning rate if the loss does not improve
    for a specified number of epochs (patience).
    """
    def __init__(self, initial_learning_rate, patience=15):
        super(LRreducer, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.patience = patience  # Number of epochs to wait before reducing LR
        self.best_loss = float('inf')
        self.wait = 0
        self.current_lr = initial_learning_rate

    def __call__(self, step):
        return self.current_lr

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['loss']
        if self.best_loss <= current_loss :
            self.wait += 1
            if self.wait >= self.patience:
                self.current_lr /= 1.3
                self.wait = 0
                print(f"\nLearning rate reduced to {self.current_lr}")
        else:
            self.best_loss = current_loss
            self.wait = 0

class printlearningrate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = tf.keras.backend.eval(optimizer.learning_rate)
        Epoch_count = epoch + 1
        print('\n', "Epoch:", Epoch_count, ', LR: {:.2e}'.format(lr))



if __name__ == "__main__":
    model_1D = model_1D_def()
    model_1D.summary()

    model_2D = model_2D_def()
    model_2D.summary()


# import visualkeras
# visualkeras.layered_view(model_1D, 
#                          legend=True, 
#                         #  draw_volume=False
#                          ).show()
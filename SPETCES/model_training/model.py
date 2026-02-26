import numpy as np 
import tensorflow as tf
# import tensorflow_addons as tfa

INIT_LR = 1e-4
MAX_LR = 1e-2
regul=0

def model_1D_def(trace_shape=512,
                 kernel_size=11,
                 CNN_layers=32
                 ):
    """
    1D Model -- taken from Sandra le Coz's work
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(trace_shape, 2)),
        tf.keras.layers.Conv1D(CNN_layers, 
                               kernel_size=(kernel_size,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(CNN_layers, 
                               kernel_size=(kernel_size,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(CNN_layers, 
                               kernel_size=(kernel_size,), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_2D_def(trace_shape=512,
                 kernel_size=11,
                 CNN_layers=32
                 ):
    """
    2D Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(trace_shape, 2, 1)),
        tf.keras.layers.Conv2D(CNN_layers, 
                               kernel_size=(kernel_size,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(CNN_layers, 
                               kernel_size=(kernel_size,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(CNN_layers, 
                               kernel_size=(kernel_size,2), 
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(regul),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ----------------- Model ResNet 1D and 2D definitions -----------------

def resnet_block_1d(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_cnn1d_resnet(input_shape=(384, 2),
                       filters=16,
                       kernel_size=3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Couche initiale
    x = tf.keras.layers.Conv1D(filters*4, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Blocs ResNet
    x = resnet_block_1d(x, filters, kernel_size=kernel_size)
    x = resnet_block_1d(x, filters*2, kernel_size=kernel_size, stride=2)
    x = resnet_block_1d(x, filters*2, kernel_size=kernel_size)
    x = resnet_block_1d(x, filters*4, kernel_size=kernel_size, stride=2)
    x = resnet_block_1d(x, filters*4, kernel_size=kernel_size)
    # Couche finale
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def resnet_block_2d(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_cnn2d_resnet(input_shape=(384, 2, 1),
                       filters=16,
                       kernel_size=3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Couche initiale
    x = tf.keras.layers.Conv2D(filters*4, (7, 1), strides=(2, 1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 1), strides=(2, 1), padding='same')(x)

    # Blocs ResNet
    x = resnet_block_2d(x, filters, kernel_size=kernel_size)
    x = resnet_block_2d(x, filters*2, kernel_size=kernel_size, stride=2)
    x = resnet_block_2d(x, filters*2, kernel_size=kernel_size)
    x = resnet_block_2d(x, filters*4, kernel_size=kernel_size, stride=2)
    x = resnet_block_2d(x, filters*4, kernel_size=kernel_size)

    # Couche finale
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


class LRreducer(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate reducer that halves the learning rate if the loss does not improve
    for a specified number of epochs (patience).
    """
    def __init__(self, initial_learning_rate, patience=15, divide=1.5):
        super(LRreducer, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.patience = patience  # Number of epochs to wait before reducing LR
        self.divide = divide # Factor to reduce the LR by
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
                self.current_lr /= self.divide
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


# In order to get the learning rate from the history object
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = self.model.optimizer.learning_rate.numpy()
        print(f"Learning rate: {logs['learning_rate']}")

if __name__ == "__main__":
    # ICRC2023 models
    model_1D = model_1D_def()
    model_1D.summary()

    model_2D = model_2D_def()
    model_2D.summary()

    # ResNet models
    model_1d_resnet = build_cnn1d_resnet(input_shape=(384, 2))
    model_1d_resnet.summary()

    model_2d_resnet = build_cnn2d_resnet(input_shape=(384, 2, 1))
    model_2d_resnet.summary()


# import visualkeras
# visualkeras.layered_view(model_1d_resnet, 
#                          legend=True, 
#                         #  draw_volume=False,
#                         to_file='model_1d_resnet.png'
#                          )#.show()
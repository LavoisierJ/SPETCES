import numpy as np 
import tensorflow as tf


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

model_1D = model_1D_def()
model_1D.summary()

model_2D = model_2D_def()
model_2D.summary()


# import visualkeras
# visualkeras.layered_view(model_1D, 
#                          legend=True, 
#                         #  draw_volume=False
#                          ).show()
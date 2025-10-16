"""

File containing all paths and graphical functions for ML training.

"""

import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.signal import periodogram

fs = 5e8 # in Hz, sampling frequency for the traces

master_path = "/Users/jolan/Documents/GRAND_Work/Pipeline_ML/machine_learning" # path to the outputs datasets and plots
classifier_path = "/Users/jolan/Documents/GRAND_Work/Pipeline_ML/machine_learning/trace_classifier_ML" # path of the main folder of the project, containing the folder sans_les_mains and trace_classifier_ML

# Paths of weights

path_weights_1D = classifier_path + "/model_training/weights_model/model_1D.h5"
path_weights_2D = classifier_path + "/model_training/weights_model/model_2D.h5"

path_weights_mine_plain_1D = classifier_path + "/model_training/weights_model/model_1D_mine_plane.h5"
path_weights_mine_plain_2D = classifier_path + "/model_training/weights_model/model_2D_mine_plane.h5"


def plot_loss(history,
              save_path,
              display=True
              ) :
    plt.plot(history.epoch, np.array(history.history['loss']),label = 'Train loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Validation loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (binary crossentropy)')
    plt.yscale('log')
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()

def plot_accuracy(history,
                  save_path,
                  display=True
                  ) :
    plt.plot(history.epoch, np.array(history.history['accuracy']),label = 'Train accuracy')
    plt.plot(history.epoch, np.array(history.history['val_accuracy']),label = 'Validation accuracy')
    plt.grid()
    plt.legend()
    plt.title(str(int(np.ceil(history.history['accuracy'][-1]*100)))+'%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()

def plot_trace(trace,
               save_path,
               title=None
               ) :
    """
    Entries:
        trace: trace to plot, shape (512, 2)
        save_path: str, path wherein to save plot
        title: str, title of the plot
    Output:
        None, saves a plot in save_path
    """
    plt.figure(figsize=(10, 6))
    for k in range(2):
        plt.plot(
                np.linspace(0,1024, 512),
                    trace[: ,k], 
                    label=f'Channel {k+1}')
    plt.title(title)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude ADC')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_trace_and_PSD(trace,
                       save_path,
                       title=None
                       ) :
    """
    Entries:
        trace: trace to plot, shape (512, 2)
        save_path: str, path wherein to save plot
        title: str, title of the plot
    Output:
        None, saves a plot of the trace and its PSD in save_path
    """
    fig, ax = plt.subplots(1, 2,
                           figsize=(14, 6))
    fig.suptitle(title)
    for k in range(2):
        ax[0].plot(
                np.linspace(0,1024, 512),
                    trace[: ,k], 
                    label=f'Channel {k+1}')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Amplitude ADC')
    ax[0].legend()
    for k in range(2):
        f, Px = periodogram(trace[:,k], fs=fs)
        ax[1].semilogy(f, Px,
             label=f'Channel {k+1}')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('PSD [ADC^2/Hz]')
    ax[1].legend()
    plt.savefig(save_path)
    plt.close()


def PSD_dataset(repertory_dataset,
                save_path
                ) :
    """
    Entries:
        repertory_dataset: str, path to the .npy files of the dataset
        save_path: str, path wherein to save plot
    Output:
        None, saves a plot of the average PSD of the dataset in save_path
    """
    data = np.load(repertory_dataset[0])
    plt.figure(figsize=(10,6))
    fx, Pxx = periodogram(data[0,:,0], fs=fs)
    fy, Pxy = periodogram(data[0,:,1], fs=fs)

    div = 0

    for i in range(len(repertory_dataset)) :
        data = np.load(repertory_dataset[i])
        for j in range(data.shape[0]):
            if i==0 and j==0 :
                continue
            _, Pxx1 = periodogram(data[j,:,0], fs=fs)
            Pxx += Pxx1
            _, Pxy1 = periodogram(data[j,:,1], fs=fs)
            Pxy += Pxy1
            div += 1

    plt.semilogy(fx, Pxx/div,
                 label='X channel')
    plt.semilogy(fy, Pxy/div,
                 label='Y channel')
    plt.xlabel('Frequency [Hz]',
               fontsize=14)
    plt.ylabel('PSD [ADC^2/Hz]',
               fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'PSD for dataset {repertory_dataset.split("/")[-1]}',
              wrap=True,
              fontsize=16)
    plt.legend(fontsize=12)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()
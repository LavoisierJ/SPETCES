"""

File containing all paths and measurement/graphical functions for ML training.

"""

import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.signal import periodogram

fs = 5e8 # in Hz, sampling frequency for the traces

master_path = "/sps/grand/jlavoisier/output/ML_cuts" # path to the outputs datasets and plots
classifier_path = "/pbs/home/j/jlavoisier/SPETCES/SPETCES" # path of the main folder of the project, containing the folder trace_classifier_ML

# Paths of weights

path_weights_1D = classifier_path + "/model_training/weights_model/"
path_weights_2D = classifier_path + "/model_training/weights_model/"

path_weights_average_1D = classifier_path + "/model_training/model_average/weights_model_1D/"
path_weights_average_2D = classifier_path + "/model_training/model_average/weights_model_2D/"

path_weights_mine_plain_1D = classifier_path + "/model_training/weights_model/model_1D_mine_plane.h5"
path_weights_mine_plain_2D = classifier_path + "/model_training/weights_model/model_2D_mine_plane.h5"


def measure_SNR(traces) :
    """
    Entries:
        traces: np.array of shape (n_traces, 512, 2)
    Output:
        SNRs: np.array of shape (n_traces, 2), SNR for each channel of each trace
    """
    n_traces = traces.shape[0]
    SNRs = np.zeros((n_traces))
    for i in range(n_traces) :
        signal_power = np.zeros(2)
        noise_power = np.zeros(2)
        for k in range(2) :
            signal_power[k] = np.max(traces[i, :, k])
            noise_power[k] = np.std(np.concatenate((traces[i, 0:np.argmax(traces[i, :, k])-10, k], traces[i, np.argmax(traces[i, :, k])+10:512, k])))
        SNRs[i] = np.max(signal_power / noise_power)
    return SNRs

# ---------- Plotting functions ----------

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
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_predict_wrt_SNR(SNR,
                         predictions,
                         true_labels,
                         save_path,
                         display=True,
                         title=None,
                         accuracy=False,
                         predict_threshold=0.5,
                         predict_mean=False
                         ) :
    """
    Entries:
        SNR: np.array of shape (n_traces,), SNR of each trace
        predictions: np.array of shape (n_traces,), prediction of the model for each trace
        true_labels: np.array of shape (n_traces,), true label of each trace
        save_path: str, path wherein to save plot
        display: bool, whether to display the plot
        title: str, title of the plot
        accuracy: bool, whether to compute and display accuracy for signal and noise
        predict_threshold: float, threshold above which a trace is considered signal
        predict_mean: bool, whether to compute and display mean prediction over SNR intervals for signal and noise
    Output:
        None, saves a plot in save_path, with 2 categories: signal and noise
    """
    plt.figure(figsize=(10,6))

    plt.scatter(SNR[true_labels==0], predictions[true_labels==0],
                label='Noise traces',
                c='tab:blue',
                alpha=0.5)
    plt.scatter(SNR[true_labels==1], predictions[true_labels==1],
                label='Signal traces',
                c='tab:orange',
                alpha=0.5)
    plt.plot([0, np.max(SNR)], 
             [predict_threshold, predict_threshold],
             color='black',
             linestyle='--',
             label='Validation threshold')
    
    # --------------------------------------------------------------------------------------
    if accuracy :
        nb_intervals = 6
        accuracy_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        accuracy_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            accuracy_signal[i] = len((predictions[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)]>=predict_threshold))/len(predictions[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            accuracy_noise[i] = len((predictions[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)]<predict_threshold))/len(predictions[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])

        for i in range(nb_intervals) :
            plt.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_noise[i],accuracy_noise[i]]),
                     c='tab:blue',
                     label='Prediction accuracy' if i==0 else "")
            plt.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_signal[i],accuracy_signal[i]]),
                     c='tab:orange',
                     label='Prediction accuracy' if i==0 else "")
            
    
    # --------------------------------------------------------------------------------------
    if predict_mean :
        nb_intervals = 6
        predict_mean_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        predict_mean_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            predict_mean_signal[i] = np.mean(predictions[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            predict_mean_noise[i] = np.mean(predictions[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])
        
        for i in range(nb_intervals) :
            plt.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_noise[i],predict_mean_noise[i]]),
                     c='blue',
                    #  linestyle='dotted',
                     label='Mean noise prediction' if i==0 else "")
            plt.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_signal[i],predict_mean_signal[i]]),
                     c='red',
                    #  linestyle='dotted',
                     label='Mean signal prediction' if i==0 else "")

    # --------------------------------------------------------------------------------------
    plt.xlabel('SNR',
               fontsize=14)
    plt.ylabel('Prediction',
               fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    if title is not None:
        plt.title(title,
                  wrap=True,
                  fontsize=16)
    else:
        plt.title(f'Predictions vs SNR',
                wrap=True,
                fontsize=16)
    plt.legend(loc='center right',
               fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()


def plot_predict_wrt_SNR_CRC(SNR,
                             predictions,
                             true_labels,
                             save_path,
                             display=True,
                             title=None,
                             predict_threshold=0.5,
                             predict_mean=False
                             ) :
    """
    Entries:
        SNR: np.array of shape (n_traces,), SNR of each trace
        predictions: np.array of shape (n_traces,), prediction of the model for each trace
        true_labels: np.array of shape (n_traces,), true label of each trace
        save_path: str, path wherein to save plot
    Output:
        None, saves a plot in save_path, with 3 categories: noise, signal, cosmic ray candidates
    """

    plt.figure(figsize=(10,6))
    plt.scatter(SNR[true_labels==0], predictions[true_labels==0],
                label='Noise traces',
                alpha=0.5)
    plt.scatter(SNR[true_labels==1], predictions[true_labels==1],
                label='Signal traces',
                alpha=0.5)
    plt.scatter(SNR[true_labels==2], predictions[true_labels==2],
                label='ICRC2025 Cosmics',
                alpha=0.5,
                color='red')
    plt.plot([0, np.max(SNR)], 
             [predict_threshold, predict_threshold],
             color='black',
             linestyle='--',
             label='Validation threshold')
    plt.xlabel('SNR',
               fontsize=14)
    plt.ylabel('Prediction',
               fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if title is not None:
        plt.title(title,
                  wrap=True,
                  fontsize=16)
    else:
        plt.title(f'Predictions vs SNR',
                wrap=True,
                fontsize=16)
    plt.legend(loc='center right',
               fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()

# Plot for averaged model predictions

def plot_predict_averaged_wrt_SNR(SNR,
                         predictions,
                         true_labels,
                         save_path,
                         display=True,
                         title=None,
                         accuracy=False,
                         predict_threshold=0.5,
                         predict_mean=False
                         ) :
    """
    Plot the mean prediction of an ensemble of models with respect to the SNR of the traces.
    Entries:
        SNR: np.array of shape (n_traces,), SNR of each trace
        predictions: np.array of shape (n_traces,-1), prediction of the model for each trace
        true_labels: np.array of shape (n_traces,), true label of each trace
        save_path: str, path wherein to save plot
        display: bool, whether to display the plot
        title: str, title of the plot
        accuracy: bool, whether to compute and display accuracy for signal and noise
        predict_threshold: float, threshold above which a trace is considered signal
        predict_mean: bool, whether to compute and display mean prediction over SNR intervals for signal and noise
    Output:
        None, saves a plot in save_path, with 2 categories: signal and noise
    """
    fig = plt.figure(figsize=(10,8))

    mean_prediction = np.mean(predictions, axis=1)
    std_prediction = np.std(predictions, axis=1)

    grid = plt.GridSpec(4, 1, hspace=0, wspace=0)
    main = fig.add_subplot(grid[0:3], 
                    #    xlim=[0, np.max(SNR)+1]
                       )
    dev = fig.add_subplot(grid[3], 
                    #   xlim=[0, np.max(SNR)+1],
                      sharex=main
                      )
    

    # --------------------------------------------------------------------------------------
    if accuracy :
        nb_intervals = 6
        accuracy_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        accuracy_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            accuracy_signal[i] = len((mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)]>=predict_threshold))/len(mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            accuracy_noise[i] = len((mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)]<predict_threshold))/len(mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])

        for i in range(nb_intervals) :
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_noise[i],accuracy_noise[i]]),
                     c='tab:blue',
                     label='Prediction accuracy' if i==0 else "")
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_signal[i],accuracy_signal[i]]),
                     c='tab:orange',
                     label='Prediction accuracy' if i==0 else "")
            
    
    # --------------------------------------------------------------------------------------
    if predict_mean :
        nb_intervals = 6
        predict_mean_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        predict_mean_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            predict_mean_signal[i] = np.mean(mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            predict_mean_noise[i] = np.mean(mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])

        for i in range(nb_intervals) :
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_noise[i],predict_mean_noise[i]]),
                     c='tab:blue',
                    #  linestyle='dotted',
                     label='Mean noise prediction' if i==0 else "")
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_signal[i],predict_mean_signal[i]]),
                     c='tab:orange',
                    #  linestyle='dotted',
                     label='Mean signal prediction' if i==0 else "")

    # --------------------------------------------------------------------------------------
    
    main.scatter(SNR[true_labels==0], mean_prediction[true_labels==0],
                label='Noise traces',
                c='tab:blue',
                alpha=0.5)
    main.scatter(SNR[true_labels==1], mean_prediction[true_labels==1],
                label='Signal traces',
                c='tab:orange',
                alpha=0.5)
    main.plot([0, np.max(SNR)], 
             [predict_threshold, predict_threshold],
             color='black',
             linestyle='--',
             label='Validation threshold')
    main.set_xlabel('SNR',
               fontsize=14)
    main.set_ylabel('Average Prediction',
               fontsize=14)
    main.tick_params(axis="y",
                     direction="in",
                     labelsize=12)
    main.grid()
    main.legend(loc='center right',
               fontsize=12)


    dev.scatter(SNR[true_labels==0], std_prediction[true_labels==0],
                # label='Noise traces',
                c='tab:blue',
                alpha=0.5)
    dev.scatter(SNR[true_labels==1], std_prediction[true_labels==1],
                # label='Signal traces',
                c='tab:orange',
                alpha=0.5)
    dev.set_ylabel(r"Std Prediction",
                  fontsize=14)
    dev.set_xlabel(r"SNR",
                  fontsize=14)
    dev.tick_params(axis="both",
                     direction="in",
                     labelsize=12)

    if title is not None:
        fig.suptitle(title,
                  wrap=True,
                  fontsize=16)
    else:
        fig.suptitle(f'Predictions vs SNR',
                wrap=True,
                fontsize=16)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()


def plot_predict_averaged_wrt_SNR_CRC(SNR,
                                      predictions,
                                      true_labels,
                                      save_path,
                                      display=True,
                                      title=None,
                                      accuracy=False,
                                      predict_threshold=0.5,
                                      predict_mean=False,
                                      origin_CRC=''
                                      ) :
    """
    Plot the mean prediction of an ensemble of models with respect to the SNR of the traces.
    Entries:
        SNR: np.array of shape (n_traces,), SNR of each trace
        predictions: np.array of shape (n_traces,-1), prediction of the model for each trace
        true_labels: np.array of shape (n_traces,), true label of each trace
        save_path: str, path wherein to save plot
        display: bool, whether to display the plot
        title: str, title of the plot
        accuracy: bool, whether to compute and display accuracy for signal and noise
        predict_threshold: float, threshold above which a trace is considered signal
        predict_mean: bool, whether to compute and display mean prediction over SNR intervals for signal and noise
        origin_CRC: str, origin of the CRC (e.g. 'ICRC2025' or 'PengXiong2025')
    Output:
        None, saves a plot in save_path, with 2 categories: signal and noise
    """
    fig = plt.figure(figsize=(10,8))

    mean_prediction = np.mean(predictions, axis=1)
    std_prediction = np.std(predictions, axis=1)

    grid = plt.GridSpec(4, 1, hspace=0, wspace=0)
    main = fig.add_subplot(grid[0:3], 
                    #    xlim=[0, np.max(SNR)+1]
                       )
    dev = fig.add_subplot(grid[3], 
                    #   xlim=[0, np.max(SNR)+1],
                      sharex=main
                      )
    

    # --------------------------------------------------------------------------------------
    if accuracy :
        nb_intervals = 6
        accuracy_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        accuracy_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            accuracy_signal[i] = len((mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)]>=predict_threshold))/len(mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            accuracy_noise[i] = len((mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)]<predict_threshold))/len(mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])

        for i in range(nb_intervals) :
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_noise[i],accuracy_noise[i]]),
                     c='tab:blue',
                     label='Prediction accuracy' if i==0 else "")
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([accuracy_signal[i],accuracy_signal[i]]),
                     c='tab:orange',
                     label='Prediction accuracy' if i==0 else "")
            
    
    # --------------------------------------------------------------------------------------
    if predict_mean :
        nb_intervals = 6
        predict_mean_signal = np.zeros(nb_intervals) # 6 intervals between SNR=4 and max_SNR
        predict_mean_noise = np.zeros(nb_intervals)
        step = (np.ceil(np.max(SNR))-4)/nb_intervals
        for i in range(nb_intervals) :
            SNR_min = 4 + i*step
            SNR_max = 4 + (i+1)*step
            predict_mean_signal[i] = np.mean(mean_prediction[(true_labels==1) & (SNR>=SNR_min) & (SNR<SNR_max)])
            predict_mean_noise[i] = np.mean(mean_prediction[(true_labels==0) & (SNR>=SNR_min) & (SNR<SNR_max)])

        for i in range(nb_intervals) :
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_noise[i],predict_mean_noise[i]]),
                     c='tab:blue',
                    #  linestyle='dotted',
                     label='Mean noise prediction' if i==0 else "")
            main.plot(np.array([4 + i*step, 4 + (i+1)*step]),
                     np.array([predict_mean_signal[i],predict_mean_signal[i]]),
                     c='tab:orange',
                    #  linestyle='dotted',
                     label='Mean signal prediction' if i==0 else "")

    # --------------------------------------------------------------------------------------
    
    main.scatter(SNR[true_labels==0], mean_prediction[true_labels==0],
                label='Noise traces',
                c='tab:blue',
                alpha=0.5)
    main.scatter(SNR[true_labels==1], mean_prediction[true_labels==1],
                label='Signal traces',
                c='tab:orange',
                alpha=0.5)
    main.scatter(SNR[true_labels==2], mean_prediction[true_labels==2],
                label=f'CR Candidates ({origin_CRC})',
                alpha=0.5,
                color='red')
    main.plot([0, np.max(SNR)], 
             [predict_threshold, predict_threshold],
             color='black',
             linestyle='--',
             label='Validation threshold')
    main.set_xlabel('SNR',
               fontsize=14)
    main.set_ylabel('Average Prediction',
               fontsize=14)
    main.tick_params(axis="y",
                     direction="in",
                     labelsize=12)
    main.grid()
    main.legend(loc='center right',
               fontsize=12)


    dev.scatter(SNR[true_labels==0], std_prediction[true_labels==0],
                # label='Noise traces',
                c='tab:blue',
                alpha=0.5)
    dev.scatter(SNR[true_labels==1], std_prediction[true_labels==1],
                # label='Signal traces',
                c='tab:orange',
                alpha=0.5)
    dev.set_ylabel(r"Std Prediction",
                  fontsize=14)
    dev.set_xlabel(r"SNR",
                  fontsize=14)
    dev.tick_params(axis="both",
                     direction="in",
                     labelsize=12)

    if title is not None:
        fig.suptitle(title,
                  wrap=True,
                  fontsize=16)
    else:
        fig.suptitle(f'Predictions vs SNR',
                wrap=True,
                fontsize=16)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()

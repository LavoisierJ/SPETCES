"""
This code reads the data created by the SPETCES_cut.py code, and applies the SPETCES algorithms to it.

It outputs the predictions of the SPETCES algorithms in two *_predictions.txt files.
The first one contains the prediction for each trace 
and the second one contains the prediction for each event, by applying an average vote on the trace predictions.

The first one will have the following structure:
    - first column: file name
    - second column: event index
    - third column: antenna index
    - fourth column: prediction (either -1 if noise or SNR not required, between 0 and 1 otherwise)

The second one will have the following structure:
    - first column: file name
    - second column: event index
    - third column: prediction (average of all valid antennas)
"""

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from SPETCES.model_training.model import model_1D_def, model_2D_def, build_cnn1d_resnet, build_cnn2d_resnet
from SPETCES.imported_fcts import plot_predict_wrt_SNR, plot_predict_wrt_SNR_CRC, confusion_matrix, accuracy_precison_recall, master_path, path_weights_1D, path_weights_2D

model_name = 'heavymodel_realCR_ANhm_4_384_1024trace'

root_file = 'GP80_20250703_020024_RUN10123_CD_20dB-GP65-Y2float-62dus-CD-100000-1.root'
# root_file = "GP80_20250701_111847_RUN10122_CD_20dB-GP65-Y2float-62dus-CD-100000-1.root"
# root_file = "GP80_20250707_065719_RUN10126_CD_20dB-GP65-Y2float-10dus-TESTRUN-CD-100000-432.root" # CR candidates here

# -------- Importing the models ---------
# model_1D = model_1D_def(trace_shape=384)
# model_2D = model_2D_def(trace_shape=384)

model_1D = build_cnn1d_resnet(input_shape=(384, 2))
model_2D = build_cnn2d_resnet(input_shape=(384, 2, 1))

model_1D.load_weights(path_weights_1D + f'model_1D_{model_name}_adam.weights.h5')
model_2D.load_weights(path_weights_2D + f'model_2D_{model_name}_adam.weights.h5')

# -------- Reading the data ---------
year = root_file.split('_')[1][:4]
month = root_file.split('_')[1][4:6]

data_file = f"/sps/grand/jlavoisier/output/ML_cuts/SPETCES_cut/{year}/{month}/{root_file}"

traces = np.load(f"{data_file}/traces.npy")
info = np.load(f"{data_file}/info.npy")

# ---------- Compile models -----------

model_1D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_2D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ---------- Selection of traces with SNR and noise level ------------

SNR_min = 4
noise_level_min = 10
noise_level_max = 20

SNR_min_X = (info[:, 5] >= SNR_min)
SNR_min_Y = (info[:, 6] >= SNR_min)
noise_level_valid_X = np.logical_and((info[:, 7] >= noise_level_min), (info[:, 7] <= noise_level_max))
noise_level_valid_Y = np.logical_and((info[:, 8] >= noise_level_min), (info[:, 8] <= noise_level_max))

valid = np.logical_and(np.logical_or(SNR_min_X, SNR_min_Y), np.logical_and(noise_level_valid_X, noise_level_valid_Y))

# ---------- Predictions -----------

prediction_1D = model_1D.predict(traces,
                                #  true_test, 
                                #  verbose=0
                                 )
prediction_2D = model_2D.predict(traces,  
                                #  true_test, 
                                #  verbose=0
                                 )


# ------------------- Writing Files -----------------------------

# Saving the predictions by antenna in a txt file

with open(f"{data_file}/{model_name}_predictions_1D.txt", "w") as f:
    f.write("# file_name\t event_index\t antenna_index\t prediction\n")
    file_name = root_file
    for i in range(prediction_1D.shape[0]):
        event_index = info[i][2]
        antenna_index = info[i][3]

        if valid[i]: # only save the predictions for the valid traces
            prediction = prediction_1D[i][0]
        else:
            prediction = -1 # if the trace is not valid, we put -1 as prediction

        f.write(f"{file_name}\t {int(event_index)}\t {int(antenna_index)}\t {prediction}\n")

with open(f"{data_file}/{model_name}_predictions_2D.txt", "w") as f:
    f.write("# file_name\t event_index\t antenna_index\t prediction\n")
    file_name = root_file
    for i in range(prediction_2D.shape[0]):
        event_index = info[i][2]
        antenna_index = info[i][3]

        if valid[i]: # only save the predictions for the valid traces
            prediction = prediction_2D[i][0]
        else:
            prediction = -1 # if the trace is not valid, we put -1 as prediction

        f.write(f"{file_name}\t {int(event_index)}\t {int(antenna_index)}\t {prediction}\n")



# Saving the predictions by event in a txt file (average of all valid antennas)
pred_per_event_1D = np.array([])
pred_per_event_2D = np.array([])

with open(f"{data_file}/{model_name}_predictions_1D_by_event.txt", "w") as f:
    f.write("# file_name\t event_index\t prediction\n")
    file_name = root_file

    for event_index in np.unique(info[:, 2]):
        valid_predictions = prediction_1D[info[:, 2] == event_index][valid[info[:, 2] == event_index]]
        if valid_predictions.size > 0:
            average_prediction = np.mean(valid_predictions)
        else:
            # if there are no valid predictions for this event, we put -1 as prediction
            average_prediction = -1
        pred_per_event_1D = np.append(pred_per_event_1D, average_prediction)
        f.write(f"{file_name}\t {int(event_index)}\t {average_prediction}\n")

with open(f"{data_file}/{model_name}_predictions_2D_by_event.txt", "w") as f:
    f.write("# file_name\t event_index\t prediction\n")
    file_name = root_file

    for event_index in np.unique(info[:, 2]):
        valid_predictions = prediction_2D[info[:, 2] == event_index][valid[info[:, 2] == event_index]]
        if valid_predictions.size > 0:
            average_prediction = np.mean(valid_predictions)
        else:
            # if there are no valid predictions for this event, we put -1 as prediction
            average_prediction = -1
        pred_per_event_2D = np.append(pred_per_event_2D, average_prediction)
        f.write(f"{file_name}\t {int(event_index)}\t {average_prediction}\n")


# ----------------------- Plots of the predictions with respect to the SNR for each antenna -----------------------


norm = mpl.colors.Normalize(vmin=0, vmax=1) 
sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
sm.set_array([]) 

plt.figure(figsize=(10, 5))
plt.suptitle(f'Predictions: {model_name}', fontsize=20)
plt.subplot(1, 2, 1)
plt.scatter(np.sqrt(info[valid, 5]**2 + info[valid, 6]**2), 
            prediction_1D[valid], 
            alpha=0.5,
            c=prediction_1D[valid],
            cmap='cool')

plt.xlabel('SNR', fontsize=16)
plt.ylabel('Prediction 1D', fontsize=16)
plt.ylim(0,1)
plt.tick_params(axis='both', labelsize=14)
plt.title('Prediction 1D vs SNR', fontsize=18)

# plt.colorbar(label='Prediction 1D', pad=0.01)
plt.colorbar(sm, 
             ax=plt.gca(), 
            #  label='Prediction 1D', 
             pad=0.01
             )

plt.subplot(1, 2, 2)
plt.scatter(np.sqrt(info[valid, 5]**2 + info[valid, 6]**2), 
            prediction_2D[valid], 
            alpha=0.5,
            c=prediction_2D[valid],
            cmap='cool')

plt.xlabel('SNR', fontsize=16)
plt.ylabel('Prediction 2D', fontsize=16)
plt.ylim(0,1)
plt.tick_params(axis='both', labelsize=14)
plt.title('Prediction 2D vs SNR', fontsize=18)
# plt.colorbar(label='Prediction 2D', pad=0.01)

plt.colorbar(sm, 
             ax=plt.gca(), 
            #  label='Prediction 2D', 
             pad=0.01
             ) 
plt.tight_layout()
plt.show()
plt.savefig(f"{data_file}/{model_name}_predictions_per_ant.png")
plt.close()

# Plot for prediction by event
SNR_per_event = np.array([np.sqrt(info[info[:, 2] == event_index, 5]**2 + info[info[:, 2] == event_index, 6]**2).mean() for event_index in np.unique(info[:, 2])])

plt.figure(figsize=(10, 5))
plt.suptitle(f'Predictions: {model_name}', fontsize=20)
plt.subplot(1, 2, 1)
mask_valid_events = (pred_per_event_1D != -1)
plt.scatter(SNR_per_event[mask_valid_events], 
            pred_per_event_1D[mask_valid_events], 
            alpha=0.5,
            c=pred_per_event_1D[mask_valid_events],
            cmap='cool')
plt.plot(np.linspace(np.min(SNR_per_event), np.max(SNR_per_event), 10),
         np.zeros(10) + 0.5,
         color='red', 
         linestyle='--', 
         label='Threshold 0.5'
         )

# for i in range(len(np.unique(info[:, 2]))):
#     if pred_per_event_1D[i] >= 0.5: # only annotate the points with a prediction above 0.5
#         plt.annotate(f"{np.unique(info[i, 2])}", 
#                     (SNR_per_event[i], pred_per_event_1D[i]),
#                     textcoords="offset points", 
#                     # xytext=(0,10), 
#                     ha='right')

plt.xlabel('SNR', fontsize=16)
plt.ylabel('Prediction 1D', fontsize=16)
plt.ylim(0,1)
plt.tick_params(axis='both', labelsize=14)
plt.title('Prediction 1D per event vs SNR', fontsize=18)
plt.colorbar(sm, 
             ax=plt.gca(), 
            #  label='Prediction 1D', 
             pad=0.01
             )

plt.subplot(1, 2, 2)
mask_valid_events = (pred_per_event_2D != -1)
plt.scatter(SNR_per_event[mask_valid_events], 
            pred_per_event_2D[mask_valid_events], 
            alpha=0.5,
            c=pred_per_event_2D[mask_valid_events],
            cmap='cool')
plt.plot(np.linspace(np.min(SNR_per_event), np.max(SNR_per_event), 10),
         np.zeros(10) + 0.5,
         color='red', 
         linestyle='--', 
         label='Threshold 0.5'
         )
# for i in range(len(np.unique(info[:, 2]))):
#     if pred_per_event_2D[i] >= 0.5: # only annotate the points with a prediction above 0.5
#         plt.annotate(f"{np.unique(info[i, 2])}", 
#                     (SNR_per_event[i], pred_per_event_2D[i]),
#                     textcoords="offset points", 
#                     # xytext=(0,10), 
#                     ha='right')
        
plt.xlabel('SNR', fontsize=16)
plt.ylabel('Prediction 2D', fontsize=16)
plt.ylim(0,1)
plt.tick_params(axis='both', labelsize=14)
plt.title('Prediction 2D per event vs SNR', fontsize=18)
cbar = plt.colorbar(sm, 
             ax=plt.gca(), 
            #  label='Prediction 2D', 
             pad=0.01
             ) 

# cbar.set_ticks(fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig(f"{data_file}/{model_name}_predictions_per_event.png")
plt.close()
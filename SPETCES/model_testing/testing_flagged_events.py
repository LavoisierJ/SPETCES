"""

Testing the trained models with respect to the SNR of the input data.

"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import measure_SNR, plot_predict_wrt_SNR, plot_predict_wrt_SNR_CRC, four_notch_filters, confusion_matrix, accuracy_precison_recall, master_path, path_weights_1D, path_weights_2D, fs

def plot_predict_flagged_wrt_SNR(SNR,
                                 predictions,
                                 true_labels,
                                 save_path,
                                 display=False,
                                 title='Model Predictions vs SNR',
                                 predict_threshold=0.5,
                                 predict_mean=False,
                                 x_label='SNR'
                                 ) :
    """
    Plot the predictions of flagged events with respect to the SNR of the traces.
    Entries:
        SNR : array, SNR of each trace
        predictions : array, model predictions for each trace
        true_labels : array, true labels for each trace
        save_path : str, path to save the plot
        display : bool, whether to display the plot or not
        title : str, title of the plot
        predict_mean : bool, whether to plot the mean prediction per SNR bin
        x_label : str, label for the x-axis
    """

    plt.figure(figsize=(10,6))
    plt.scatter(SNR[true_labels==0], 
                predictions[true_labels==0], 
                # color='blue', 
                alpha=0.5, 
                label='Traces of flagged_events')

    plt.scatter(SNR[true_labels==1], 
                predictions[true_labels==1],
                label='Traces of CRC',
                alpha=0.5)

    plt.plot([0, np.max(SNR)], 
             [predict_threshold, predict_threshold],
             color='black',
             linestyle='--',
             label='Validation threshold')

    # -------------------------------------
    if predict_mean :
        # Binning
        bins = np.linspace(np.min(SNR), np.max(SNR), 10)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mean_predictions = []
        for j in range(len(bins)-1) :
            indices = np.where((SNR >= bins[j]) & (SNR < bins[j+1]))[0]
            if len(indices) > 0 :
                mean_pred = np.mean(predictions[indices])
            else :
                mean_pred = np.nan
            mean_predictions.append(mean_pred)
        plt.plot(bin_centers, mean_predictions, color='black', marker='o', linestyle='-', label='Mean Prediction')

    # -------------------------------------

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Model Prediction', fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=14)
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.savefig(save_path)
    if display :
        plt.show()
    plt.close()


dataset_name = 'RUNTEST_2'

divide_mean_noise = 15
# -------- Importing the models ---------
model_1D = model_1D_def()
model_2D = model_2D_def()
model_1D.load_weights(path_weights_1D + f'model_1D_{dataset_name}_adam.weights.h5')
model_2D.load_weights(path_weights_2D + f'model_2D_{dataset_name}_adam.weights.h5')

# # ---------- Importing test data -----------

# # data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
# # data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

# data_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/test_dataset_RUNTEST_2_3566_traces.npy')
# info_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/test_dataset_RUNTEST_2_3566_info.npy')
# # data_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/test_dataset_RUNTEST_2_1_ANhm_3889_traces.npy')
# # info_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/test_dataset_RUNTEST_2_1_ANhm_3889_info.npy')

# data_signal_test = np.empty((0,512,2))  # No signal data for this test
# info_signal_test = np.empty((0,11))

# for i in range(info_noise_test.shape[0]) :
#     info_noise_test[i,0] = i  # Re-indexing the trace IDs

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

# # # Shuffle
# # liste_test = np.arange(number_data_signal_test + number_data_noise_test) 
# # np.random.shuffle(liste_test)
# # data_test = data_test[liste_test]
# # true_test = true_test[liste_test]


# # ---------- Compile models -----------

# model_1D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model_2D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# # ---------- Predictions -----------

# prediction_1D = model_1D.predict(data_test,
#                                 #  true_test, 
#                                 #  verbose=0
#                                  )
# prediction_2D = model_2D.predict(data_test,  
#                                 #  true_test, 
#                                 #  verbose=0
#                                  )

# print(prediction_1D)
# # SNR_test = measure_SNR(data_test)
# SNR_test = np.max(info_noise_test[:,5:7], axis=1)  # SNR is already measured and stored in the info file
# SNR_test = np.append(SNR_test, np.max(info_signal_test[:,5:7], axis=1), axis=0)
# print("SNR measured.")

# plot_predict_wrt_SNR(SNR_test,
#                      prediction_1D,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_SNR',
#                      display=False,
#                      title='Model 1D Predictions vs SNR',
#                      predict_mean=True)

# plot_predict_wrt_SNR(SNR_test,
#                      prediction_2D,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_SNR',
#                      display=False,
#                      title='Model 2D Predictions vs SNR',
#                      predict_mean=True)

# sigma_test = np.max(info_noise_test[:,7:9], axis=1)
# sigma_test = np.append(sigma_test, np.max(info_signal_test[:,7:9], axis=1), axis=0)

# plot_predict_wrt_SNR(sigma_test,
#                      prediction_1D,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_sigma',
#                      display=False,
#                      title='Model 1D Predictions vs Bckg noise level',
#                      predict_mean=True,
#                      x_label='Background noise level (std dev)')

# plot_predict_wrt_SNR(sigma_test,
#                      prediction_2D,
#                      true_labels=true_test,
#                      save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_sigma',
#                      display=False,
#                      title='Model 2D Predictions vs Bckg noise level',
#                      predict_mean=True,
#                      x_label='Background noise level (std dev)')

# # confusion matrix
# cm_1D = confusion_matrix(prediction_1D, true_test)
# cm_2D = confusion_matrix(prediction_2D, true_test)

# print("Confusion Matrix Model 1D: TP=", cm_1D[1,1], " TN=", cm_1D[0,0], " FP=", cm_1D[0,1], " FN=", cm_1D[1,0])
# print("Confusion Matrix Model 2D: TP=", cm_2D[1,1], " TN=", cm_2D[0,0], " FP=", cm_2D[0,1], " FN=", cm_2D[1,0])

# accuracy_1D, precision_1D, recall_1D = accuracy_precison_recall(cm_1D)
# accuracy_2D, precision_2D, recall_2D = accuracy_precison_recall(cm_2D)

# print("Model 1D - Accuracy: ", accuracy_1D, " Precision: ", precision_1D, " Recall: ", recall_1D)
# print("Model 2D - Accuracy: ", accuracy_2D, " Precision: ", precision_2D, " Recall: ", recall_2D)

# # ------------ Per event -------------------
# event_ids = np.unique(info_noise_test[:, 1:3], axis=0)
# prediction_1D_event = np.array([])
# SNR_event = np.array([])
# true_test_event = np.array([])
# print(info_noise_test.shape[0])

# for event_id in event_ids :
#     indices_event = np.int32(info_noise_test[(info_noise_test[:,1] == event_id[0]) & (info_noise_test[:,2] == event_id[1])][:,0])

#     preds_event = prediction_1D[indices_event]
#     mean_pred_event = np.mean(preds_event)
#     prediction_1D_event = np.append(prediction_1D_event, mean_pred_event)
#     SNR_event = np.append(SNR_event, np.mean(info_noise_test[indices_event,5:7]))
#     true_test_event = np.append(true_test_event, true_test[indices_event][0]) # all labels are the same for the event

# # print(prediction_1D_event)
# print("Number of events: ", len(prediction_1D_event))
# print("Mean prediction 1D for events: ", np.mean(prediction_1D_event))
# print("Standard deviation prediction 1D for events: ", np.std(prediction_1D_event))

# plot_predict_wrt_SNR(SNR_event,
#                      prediction_1D_event,
#                      true_labels=true_test_event,
#                      save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_event_wrt_SNR',
#                      display=False,
#                      title='Model 1D Predictions for full events vs SNR',
#                      predict_mean=True)

true_test = np.array([])

repert_predict = sorted(glob(f'{master_path}/datasets/dataset_flagged/2025/07/*_traces.npy'))

pred_flagged_1D = np.array([])
pred_flagged_2D = np.array([])

pred_flagged_1D_event = np.array([])

for i in range(len(repert_predict)) :
    data_test = np.load(repert_predict[i])
    info_test = np.load(repert_predict[i].replace('_traces.npy', '_info.npy'))

    name = repert_predict[i].split('/')[-1]

    pred_1D = model_1D.predict(data_test)
    pred_2D = model_2D.predict(data_test)

    pred_flagged_1D = np.append(pred_flagged_1D, pred_1D)
    pred_flagged_2D = np.append(pred_flagged_2D, pred_2D)

    SNR_flagged = np.max(info_test[:,5:7], axis=1)
    SNR_test = np.append(SNR_test, SNR_flagged, axis=0)
    sigma_test = np.append(sigma_test, np.max(info_test[:,7:9], axis=1), axis=0)

    # For the 'per event' plots

    mean_pred_event = np.mean(pred_1D)
    pred_flagged_1D_event = np.append(pred_flagged_1D_event, mean_pred_event)
    SNR_event = np.append(SNR_event, np.mean(SNR_flagged))
    true_test_event = np.append(true_test_event, 0) # all labels are flagged

true_flagged = np.zeros(len(pred_flagged_1D))  # label 0 for flagged events
true_test = np.append(true_test,
                      true_flagged,
                      axis=0)

# ---------- Add predictions over cosmic ray candidates -----------
repert_predict = sorted(glob(f'{master_path}/datasets/dataset_verif/marion_202511/*_traces.npy'))

pred_CRC_1D = np.array([])
pred_CRC_2D = np.array([])

pred_CRC_1D_event = np.array([])

for i in range(len(repert_predict)) :
    data_test = np.load(repert_predict[i])
    info_test = np.load(repert_predict[i].replace('_traces.npy', '_info.npy'))

    name = repert_predict[i].split('/')[-1]

    pred_1D = model_1D.predict(data_test)
    pred_2D = model_2D.predict(data_test)

    pred_CRC_1D = np.append(pred_CRC_1D, pred_1D)
    pred_CRC_2D = np.append(pred_CRC_2D, pred_2D)

    SNR_CRC = np.max(info_test[:,5:7], axis=1)
    SNR_test = np.append(SNR_test, SNR_CRC, axis=0)
    sigma_test = np.append(sigma_test, np.max(info_test[:,7:9], axis=1), axis=0)

    # For the 'per event' plots

    mean_pred_event = np.mean(pred_1D)
    pred_CRC_1D_event = np.append(pred_CRC_1D_event, mean_pred_event)
    SNR_event = np.append(SNR_event, np.mean(SNR_CRC))
    true_test_event = np.append(true_test_event, 1) # all labels are CRC

true_CRC = np.zeros(len(pred_CRC_1D)) + 1 # label 1 for CRC


true_test = np.append(true_test,
                      true_CRC,
                      axis=0)

print(pred_CRC_1D)
print("Mean prediction CRC 1D: ", np.mean(pred_CRC_1D))
print("Standard deviation prediction CRC 1D: ", np.std(pred_CRC_1D))
print("-----")
print(pred_CRC_2D)
print("Mean prediction CRC 2D: ", np.mean(pred_CRC_2D))
print("Standard deviation prediction CRC 2D: ", np.std(pred_CRC_2D))
pred_CRC_1D = pred_CRC_1D.reshape(-1,1)
pred_CRC_2D = pred_CRC_2D.reshape(-1,1)

prediction_1D =  np.append(pred_flagged_1D, pred_CRC_1D, axis=0)
prediction_2D =  np.append(pred_flagged_2D, pred_CRC_2D, axis=0)

plot_predict_wrt_SNR_CRC(SNR_test,
                         prediction_1D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_SNR_CRC',
                         display=False,
                         title='Model 1D Predictions vs SNR (including CRC)',
                         predict_mean=True,
                         nb_intervals=8)

plot_predict_wrt_SNR_CRC(SNR_test,
                         prediction_2D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_SNR_CRC',
                         display=False,
                         title='Model 2D Predictions vs SNR (including CRC)',
                         predict_mean=True,
                         nb_intervals=8)

plot_predict_wrt_SNR_CRC(sigma_test,
                         prediction_1D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_sigma_CRC',
                         display=False,
                         title='Model 1D Predictions vs Bckg level (including CRC)',
                         predict_mean=True,
                         nb_intervals=8,
                         x_label='Background noise level (std dev)')

plot_predict_wrt_SNR_CRC(sigma_test,
                         prediction_2D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_sigma_CRC',
                         display=False,
                         title='Model 2D Predictions vs Bckg level (including CRC)',
                         predict_mean=True,
                         nb_intervals=8,
                         x_label='Background noise level (std dev)')

# ------------ Per event -------------------

prediction_1D_event = np.append(prediction_1D_event, pred_CRC_1D_event)
print("-----")
print(pred_CRC_1D_event)
print("Mean prediction CRC 1D for events: ", np.mean(pred_CRC_1D_event))
print("Standard deviation prediction CRC 1D for events: ", np.std(pred_CRC_1D_event))


plot_predict_wrt_SNR_CRC(SNR_event,
                        prediction_1D_event,
                        true_labels=true_test_event,
                        save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_event_wrt_SNR_CRC',
                        display=False,
                        title='Model 1D Predictions for full events vs SNR (including CRC)',
                        predict_mean=True)
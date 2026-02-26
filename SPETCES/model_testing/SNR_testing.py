"""

Testing the trained models with respect to the SNR of the input data.

"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import os

from SPETCES.model_training.model import model_1D_def, model_2D_def, build_cnn1d_resnet, build_cnn2d_resnet
from SPETCES.imported_fcts import measure_SNR, plot_loghist, plot_predict_wrt_SNR, plot_predict_wrt_SNR_CRC, four_notch_filters, confusion_matrix, accuracy_precison_recall, binary_crossentropy, master_path, path_weights_1D, path_weights_2D, fs


dataset_name = 'heavymodel_ANhm_6_384_1024trace'

divide_mean_noise = 15
# -------- Importing the models ---------
# model_1D = model_1D_def(trace_shape=384)
# model_2D = model_2D_def(trace_shape=384)

model_1D = build_cnn1d_resnet(input_shape=(384, 2))
model_2D = build_cnn2d_resnet(input_shape=(384, 2, 1))

model_1D.load_weights(path_weights_1D + f'model_1D_{dataset_name}_adam.weights.h5')
model_2D.load_weights(path_weights_2D + f'model_2D_{dataset_name}_adam.weights.h5')

# ---------- Importing test data -----------

# data_noise_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/prelim_signal_noise/train_test_datasets/noise_dataset_test_200_traces_noise15_SNR4.npy')
# data_signal_test = np.load(f'{master_path}/datasets/dataset_preliminary_work/dataset_1k_handmadeAN/train_test_datasets/simu_dataset_test_200_traces_noise15_SNR4.npy')

data_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/noise_dataset_test_1249_traces_noise15_SNR4_traces.npy')#[:800]
info_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/noise_dataset_test_1249_traces_noise15_SNR4_info.npy')#[:800]

data_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/ANhm_dataset_test_1260_traces_noise15_SNR4_traces.npy')#[:800]
info_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/ANhm_dataset_test_1260_traces_noise15_SNR4_info.npy')#[:800]

# data_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/marion_202512_sure_567_traces.npy')
# info_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/marion_202512_sure_567_info.npy')

# data_signal_test = np.empty((0,512,2))  # No signal data for this test
# info_signal_test = np.empty((0,11))

# # -------------------- If you want to try on the training data (to check for overfitting) ----------------------

# data_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/noise_dataset_train_2497_traces_noise15_SNR4_traces.npy')[1980:]
# info_noise_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/noise_dataset_train_2497_traces_noise15_SNR4_info.npy')[1980:]

# data_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/ANhm_dataset_train_2143_traces_noise15_SNR4_traces.npy')[:1929]
# info_signal_test = np.load(f'{master_path}/datasets/{dataset_name}/train_test_datasets/ANhm_dataset_train_2143_traces_noise15_SNR4_info.npy')[:1929]


for i in range(info_noise_test.shape[0]) :
    info_noise_test[i,0] = i  # Re-indexing the trace IDs


number_data_noise_test = np.shape(data_noise_test)[0]
number_data_signal_test = np.shape(data_signal_test)[0]

true_noise_test = np.zeros(number_data_noise_test)
true_signal_test = np.zeros(number_data_signal_test) + 1

data_test = np.append(data_noise_test, 
                      data_signal_test, 
                      axis=0)
true_test = np.append(true_noise_test,
                      true_signal_test,
                      axis=0)

# # Shuffle
# liste_test = np.arange(number_data_signal_test + number_data_noise_test) 
# np.random.shuffle(liste_test)
# data_test = data_test[liste_test]
# true_test = true_test[liste_test]


# ---------- Compile models -----------

model_1D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_2D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ---------- Predictions -----------

prediction_1D = model_1D.predict(data_test,
                                #  true_test, 
                                #  verbose=0
                                 )
prediction_2D = model_2D.predict(data_test,  
                                #  true_test, 
                                #  verbose=0
                                 )


# # -------------- In case of train data, uncomment this: ------------------
# loss_1D = binary_crossentropy(true_test, prediction_1D[:,0])
# loss_2D = binary_crossentropy(true_test, prediction_2D[:,0])


# plt.figure(figsize=(8,6))
# # plot_loghist(loss_1D,
# #              bins=50,
# #              alpha=0.3,
# #              label='All'
# #              )
# # plot_loghist(loss_1D[true_test == 0],
# #              bins=50,
# #              alpha=0.5,
# #              label='Noise'
# #              )
# # plot_loghist(loss_1D[true_test == 1],
# #              bins=50,
# #              alpha=0.5,
# #              label='Signal'
# #              )
# plt.hist(loss_1D[true_test == 0], 
#          bins=50, 
#          alpha=0.5, 
#          range=(np.min(loss_1D), np.max(loss_1D)),
#          label=f'Noise: high loss:{np.sum(loss_1D[true_test == 0] > 0.5)}, FP:{np.sum(prediction_1D[:,0][true_test == 0]>0.5)}'
#          )
# plt.hist(loss_1D[true_test == 1], 
#          bins=50, 
#          alpha=0.5, 
#          range=(np.min(loss_1D), np.max(loss_1D)), 
#          label=f'Signal: high loss:{np.sum(loss_1D[true_test == 1] > 0.5)}, FN:{np.sum(1-prediction_1D[:,0][true_test == 1]>0.5)}'
#          )

# # plt.hist(prediction_1D[:,0], bins=50, alpha=0.5, label='Model 1D Predictions')
# plt.yscale('log')

# plt.xlabel('Binary Cross-Entropy Loss', fontsize=16)
# plt.ylabel('Number of traces', fontsize=16)
# plt.title('Distribution of Binary Cross-Entropy Loss for Model 1D', fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig(f'{master_path}/plots/{dataset_name}/model1D_loss_distribution.pdf')
# plt.close()


# bad_signal_traces = np.where((loss_1D > 0.5) & (true_test == 1))[0]

# for e in bad_signal_traces :
#     plt.plot(np.linspace(0,384*2, 384), 
#              data_test[e,:,0],
#              label='X channel',
#              color='tab:blue'
#          )
#     plt.plot(np.linspace(0,384*2, 384), data_test[e,:,1],
#              label='Y channel',
#              color='tab:orange'
#              )
#     plt.title(f'Example of a misclassified signal trace (loss={loss_1D[e]:.2f})')
#     plt.xlabel('Time [ns]', fontsize=16)
#     plt.ylabel('ADC counts', fontsize=16)
#     plt.legend(fontsize=14)
#     plt.savefig(f'{master_path}/plots/{dataset_name}/traces/misclassified_signal_trace_{e}.pdf')
#     plt.close()

# bad_noise_traces = np.where((loss_1D > 0.5) & (true_test == 0))[0]

# for e in bad_noise_traces :
#     plt.plot(np.linspace(0,384*2, 384), 
#              data_test[e,:,0],
#              label='X channel',
#              color='tab:blue'
#          )
#     plt.plot(np.linspace(0,384*2, 384), data_test[e,:,1],
#              label='Y channel',
#              color='tab:orange'
#              )
#     plt.title(f'Example of a misclassified noise trace (loss={loss_1D[e]:.2f})')
#     plt.xlabel('Time [ns]', fontsize=16)
#     plt.ylabel('ADC counts', fontsize=16)
#     plt.legend(fontsize=14)
#     plt.savefig(f'{master_path}/plots/{dataset_name}/traces/misclassified_noise_trace_{e}.pdf')
#     plt.close()

# # ------------ end of training analysis -------------------



# print(prediction_1D)
# SNR_test = measure_SNR(data_test)
SNR_test = np.max(info_noise_test[:,5:7], axis=1)  # SNR is already measured and stored in the info file
SNR_test = np.append(SNR_test, np.max(info_signal_test[:,5:7], axis=1), axis=0)

# print("SNR measured.")
# print(f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_SNR')

plot_predict_wrt_SNR(SNR_test,
                     prediction_1D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_SNR.pdf',
                     display=False,
                     title='Model 1D Predictions vs SNR',
                     predict_mean=True,
                    zoom_SNR=True,
                    zoom_window=(0,70)
                    )

plot_predict_wrt_SNR(SNR_test,
                     prediction_2D,
                     true_labels=true_test,
                     save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_SNR.pdf',
                     display=False,
                     title='Model 2D Predictions vs SNR',
                     predict_mean=True,
                    zoom_SNR=True,
                    zoom_window=(0,70))

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

# confusion matrix
cm_1D = confusion_matrix(prediction_1D, true_test)
cm_2D = confusion_matrix(prediction_2D, true_test)
accuracy_1D, precision_1D, recall_1D = accuracy_precison_recall(cm_1D)
accuracy_2D, precision_2D, recall_2D = accuracy_precison_recall(cm_2D)

print("Confusion Matrix Model 1D: TP=", cm_1D[1,1], " TN=", cm_1D[0,0], " FP=", cm_1D[0,1], " FN=", cm_1D[1,0])
print("Model 1D - Accuracy: ", accuracy_1D, " Precision: ", precision_1D, " Recall: ", recall_1D)

print('-----------------------------------')

print("Confusion Matrix Model 2D: TP=", cm_2D[1,1], " TN=", cm_2D[0,0], " FP=", cm_2D[0,1], " FN=", cm_2D[1,0])
print("Model 2D - Accuracy: ", accuracy_2D, " Precision: ", precision_2D, " Recall: ", recall_2D)

# # ------------ Per event -------------------
print('----------------- By event ---------------------')

event_ids = np.unique(info_noise_test[:, 1:3], axis=0)
prediction_1D_event = np.array([])
prediction_2D_event = np.array([])
SNR_event = np.array([])
true_test_event = np.array([])
# print(info_noise_test.shape[0])
# print(np.max(info_noise_test[:,0]))

for event_id in event_ids :
    indices_event = np.int32(info_noise_test[(info_noise_test[:,1] == event_id[0]) & (info_noise_test[:,2] == event_id[1])][:,0])
    preds_event_1D = prediction_1D[indices_event]
    mean_pred_event_1D = np.mean(preds_event_1D)
    prediction_1D_event = np.append(prediction_1D_event, mean_pred_event_1D)

    preds_event_2D = prediction_2D[indices_event]
    mean_pred_event_2D = np.mean(preds_event_2D)
    prediction_2D_event = np.append(prediction_2D_event, mean_pred_event_2D)

    SNR_event = np.append(SNR_event, np.mean(info_noise_test[indices_event,5:7]))
    true_test_event = np.append(true_test_event, true_test[indices_event][0]) # all labels are the same for the event

for i in range (info_signal_test.shape[0]) :
    info_signal_test[i,0] = i  # Re-indexing the trace IDs

event_ids_signal = np.unique(info_signal_test[:, 1:3], axis=0)
for event_id in event_ids_signal :
    indices_event = np.int32(info_signal_test[(info_signal_test[:,1] == event_id[0]) & (info_signal_test[:,2] == event_id[1])][:,0])
    preds_event_1D = prediction_1D[indices_event + number_data_noise_test]
    mean_pred_event_1D = np.mean(preds_event_1D)
    prediction_1D_event = np.append(prediction_1D_event, mean_pred_event_1D)

    preds_event_2D = prediction_2D[indices_event + number_data_noise_test]
    mean_pred_event_2D = np.mean(preds_event_2D)
    prediction_2D_event = np.append(prediction_2D_event, mean_pred_event_2D)

    SNR_event = np.append(SNR_event, np.mean(info_signal_test[indices_event,5:7]))
    true_test_event = np.append(true_test_event, true_test[indices_event + number_data_noise_test][0]) # all labels are the same for the event

# print(prediction_1D_event)
print("Number of events: ", len(prediction_1D_event))
print('Number of true signal events: ', np.sum(true_test_event))
print('Number of true noise events: ', len(true_test_event) - np.sum(true_test_event))
print("-----")
print("Mean prediction 1D for signal events: ", np.mean(prediction_1D_event[true_test_event == 1]))
print("Mean prediction 1D for noise events: ", np.mean(prediction_1D_event[true_test_event == 0]))
# print("Standard deviation prediction 1D for events: ", np.std(prediction_1D_event))

cm_event_1D = confusion_matrix(prediction_1D_event, true_test_event)
# cm_2D = confusion_matrix(prediction_2D, true_test)

print("Confusion Matrix Model 1D: TP=", cm_event_1D[1,1], " TN=", cm_event_1D[0,0], " FP=", cm_event_1D[0,1], " FN=", cm_event_1D[1,0])
# print("Confusion Matrix Model 2D: TP=", cm_2D[1,1], " TN=", cm_2D[0,0], " FP=", cm_2D[0,1], " FN=", cm_2D[1,0])

accuracy_event_1D, precision_event_1D, recall_event_1D = accuracy_precison_recall(cm_event_1D)
# accuracy_2D, precision_2D, recall_2D = accuracy_precison_recall(cm_2D)

print("Model 1D - Accuracy: ", accuracy_event_1D, " Precision: ", precision_event_1D, " Recall: ", recall_event_1D)

plot_predict_wrt_SNR(SNR_event,
                     prediction_1D_event,
                     true_labels=true_test_event,
                     save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_event_wrt_SNR.pdf',
                     display=False,
                     title='Model 1D Predictions for full events vs SNR',
                     predict_mean=True)


# print(prediction_2D_event)
print("Number of events: ", len(prediction_2D_event))
print('Number of true signal events: ', np.sum(true_test_event))
print('Number of true noise events: ', len(true_test_event) - np.sum(true_test_event))
print("-----")
print("Mean prediction 2D for signal events: ", np.mean(prediction_2D_event[true_test_event == 1]))
print("Mean prediction 2D for noise events: ", np.mean(prediction_2D_event[true_test_event == 0]))
# print("Standard deviation prediction 1D for events: ", np.std(prediction_1D_event))

cm_event_2D = confusion_matrix(prediction_2D_event, true_test_event)
# cm_2D = confusion_matrix(prediction_2D, true_test)

print("Confusion Matrix Model 2D: TP=", cm_event_2D[1,1], " TN=", cm_event_2D[0,0], " FP=", cm_event_2D[0,1], " FN=", cm_event_2D[1,0])

accuracy_event_2D, precision_event_2D, recall_event_2D = accuracy_precison_recall(cm_event_2D)

print("Model 2D - Accuracy: ", accuracy_event_2D, " Precision: ", precision_event_2D, " Recall: ", recall_event_2D)

plot_predict_wrt_SNR(SNR_event,
                     prediction_2D_event,
                     true_labels=true_test_event,
                     save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_event_wrt_SNR.pdf',
                     display=False,
                     title='Model 2D Predictions for full events vs SNR',
                     predict_mean=True)




# ---------- Add predictions over cosmic ray candidates -----------
repert_predict = sorted(glob(f'{master_path}/datasets/dataset_verif/marion_202512_sure/traces.npy'))
# repert_predict = sorted(glob(f'{master_path}/datasets/heavymodel_realCR_ANhm_5_384_1024trace/train_test_datasets/CR_test_dataset_364_traces_noise15_SNR4_traces.npy'))

pred_CRC_1D = np.array([])
pred_CRC_2D = np.array([])

pred_CRC_1D_event = np.array([])
pred_CRC_2D_event = np.array([])

file_index_CRC = np.array([])
event_index_CRC = np.array([])

for i in range(len(repert_predict)) :
    data_test = np.load(repert_predict[i])
    info_test = np.load(repert_predict[i].replace('traces.npy', 'info.npy'))

    # name = repert_predict[i].split('/')[-1].replace('_traces.npy', '')

    # If you want to apply filtering to the traces before prediction
    # filtered_data = np.zeros_like(data_test)
    # # Normalization individually
    # for j in range(data_test.shape[0]) :
    #     filtered_data[j, :, 0] = four_notch_filters(data_test[j, :, 0], fs)/info_test[j,7]
    #     filtered_data[j, :, 1] = four_notch_filters(data_test[j, :, 1], fs)/info_test[j,8]
    # data_test = filtered_data
    # for j in range(data_test.shape[0]) :
    #     plt.plot(np.linspace(0,1024, 512), 
    #              data_test[j,:,0],
    #              label='X channel',
    #              color='tab:blue'
    #          )
    #     plt.plot(np.linspace(0,1024, 512), data_test[j,:,1],
    #             label='Y channel',
    #             color='tab:orange'
    #             )
    #     plt.title(f'Example trace from {name} after filtering')
    #     plt.xlabel('Time [ns]', fontsize=16)
    #     plt.ylabel('ADC counts', fontsize=16)
    #     plt.legend(fontsize=14)
    #     plt.savefig(f'{master_path}/plots/{dataset_name}/trace/example_trace_{name}_{j}_filtered.png')
    #     plt.close()

    pred_1D = model_1D.predict(data_test)
    pred_2D = model_2D.predict(data_test)

    good_pred_1D = pred_1D[:,0] > 0.5
    # good_pred_2D = pred_2D[:,0] > 0.5

    bad_pred_1D = pred_1D[:,0] <= 0.5
    # bad_pred_2D = pred_2D[:,0] <= 0.5

    good_data_1D = data_test[good_pred_1D]
    good_info_1D = info_test[good_pred_1D]
    # good_data_2D = data_test[good_pred_2D]
    # good_info_2D = info_test[good_pred_2D]

    # for j in range(good_data_1D.shape[0]) :
    #     plt.plot(np.linspace(0,1024, 512), 
    #              good_data_1D[j,:,0],
    #              label='X channel',
    #              color='tab:blue'
    #          )
    #     plt.plot(np.linspace(0,1024, 512), good_data_1D[j,:,1],
    #             label='Y channel',
    #             color='tab:orange'
    #             )
    #     plt.title(f'Example trace from {name} after filtering')
    #     plt.xlabel('Time [ns]', fontsize=16)
    #     plt.ylabel('ADC counts', fontsize=16)
    #     plt.legend(fontsize=14)
    #     plt.savefig(f'{master_path}/plots/{dataset_name}/predictions/good/trace_{name}_{j}_{pred_1D[good_pred_1D][j]}.png')
    #     plt.close()
    
    bad_data_1D = data_test[bad_pred_1D]
    bad_info_1D = info_test[bad_pred_1D]
    # for j in range(bad_data_1D.shape[0]) :
    #     plt.plot(np.linspace(0,1024, 512), 
    #              bad_data_1D[j,:,0],
    #              label='X channel',
    #              color='tab:blue'
    #          )
    #     plt.plot(np.linspace(0,1024, 512), bad_data_1D[j,:,1],
    #             label='Y channel',
    #             color='tab:orange'
    #             )
    #     plt.title(f'Example trace from {name} after filtering')
    #     plt.xlabel('Time [ns]', fontsize=16)
    #     plt.ylabel('ADC counts', fontsize=16)
    #     plt.legend(fontsize=14)
    #     plt.savefig(f'{master_path}/plots/{dataset_name}/predictions/bad/trace_{name}_{j}_{pred_1D[bad_pred_1D][j]}.png')
    #     plt.close()


    pred_CRC_1D = np.append(pred_CRC_1D, pred_1D)
    pred_CRC_2D = np.append(pred_CRC_2D, pred_2D)

    SNR_CRC = np.max(info_test[:,5:7], axis=1)
    SNR_test = np.append(SNR_test, SNR_CRC, axis=0)
    # sigma_test = np.append(sigma_test, np.max(info_test[:,7:9], axis=1), axis=0)

    # # For the 'per event' plots

    #For list of events
    # mean_pred_event = np.mean(pred_1D)
    # pred_CRC_1D_event = np.append(pred_CRC_1D_event, mean_pred_event)
    # mean_pred_event = np.mean(pred_2D)
    # pred_CRC_2D_event = np.append(pred_CRC_2D_event, mean_pred_event)

    # SNR_event = np.append(SNR_event, np.mean(SNR_CRC))
    # true_test_event = np.append(true_test_event, 2) # all labels are CRC

    # For all traces in a single .npy file


    event_ids_signal = np.unique(info_test[:, 1:3], axis=0)
    for event_id in event_ids_signal :
        indices_event = np.int32(info_test[(info_test[:,1] == event_id[0]) & (info_test[:,2] == event_id[1])][:,0])
        file_index = info_test[(info_test[:,1] == event_id[0]) & (info_test[:,2] == event_id[1])][0,1]
        event_index = info_test[(info_test[:,1] == event_id[0]) & (info_test[:,2] == event_id[1])][0,2]
        print(f'Event ID: {event_id}, file index: {file_index}, event index: {event_index}, number of traces in event: {len(indices_event)}')

        preds_event_1D = pred_CRC_1D[indices_event]
        mean_pred_event_1D = np.mean(preds_event_1D)
        pred_CRC_1D_event = np.append(pred_CRC_1D_event, mean_pred_event_1D)

        preds_event_2D = pred_CRC_2D[indices_event]
        mean_pred_event_2D = np.mean(preds_event_2D)
        pred_CRC_2D_event = np.append(pred_CRC_2D_event, mean_pred_event_2D)

        SNR_event = np.append(SNR_event, np.mean(info_test[indices_event,5:7]))

        file_index_CRC = np.append(file_index_CRC, file_index)
        event_index_CRC = np.append(event_index_CRC, event_index)
        # true_test_event = np.append(true_test_event, true_test[indices_event + number_data_noise_test + number_data_signal_test][0]) # all labels are the same for the event



true_CRC = np.zeros(len(pred_CRC_1D)) + 2 # label 2 for CRC


true_test = np.append(true_test,
                      true_CRC,
                      axis=0)

cm_CRC_1D = confusion_matrix(pred_CRC_1D, np.zeros(len(pred_CRC_1D)) + 1)
cm_CRC_2D = confusion_matrix(pred_CRC_2D, np.zeros(len(pred_CRC_2D)) + 1)
accuracy_CRC_1D, precision_CRC_1D, recall_CRC_1D = accuracy_precison_recall(cm_CRC_1D)
accuracy_CRC_2D, precision_CRC_2D, recall_CRC_2D = accuracy_precison_recall(cm_CRC_2D)

# print(pred_CRC_1D)
print("------------- test on CR candidates -------------")
print("Mean prediction CRC 1D: ", np.mean(pred_CRC_1D))
print("Standard deviation prediction CRC 1D: ", np.std(pred_CRC_1D))

print("Confusion Matrix Model 1D: TP=", cm_CRC_1D[1,1], " TN=", cm_CRC_1D[0,0], " FP=", cm_CRC_1D[0,1], " FN=", cm_CRC_1D[1,0])
print("Model 1D - Accuracy: ", accuracy_CRC_1D, " Precision: ", precision_CRC_1D, " Recall: ", recall_CRC_1D)

print("----------------------------")
# print(pred_CRC_2D)
print("Mean prediction CRC 2D: ", np.mean(pred_CRC_2D))
print("Standard deviation prediction CRC 2D: ", np.std(pred_CRC_2D))
print("Confusion Matrix Model 2D: TP=", cm_CRC_2D[1,1], " TN=", cm_CRC_2D[0,0], " FP=", cm_CRC_2D[0,1], " FN=", cm_CRC_2D[1,0])
print("Model 2D - Accuracy: ", accuracy_CRC_2D, " Precision: ", precision_CRC_2D, " Recall: ", recall_CRC_2D)

pred_CRC_1D = pred_CRC_1D.reshape(-1,1)
pred_CRC_2D = pred_CRC_2D.reshape(-1,1)

prediction_1D =  np.append(prediction_1D, pred_CRC_1D, axis=0)
prediction_2D =  np.append(prediction_2D, pred_CRC_2D, axis=0)

plot_predict_wrt_SNR_CRC(SNR_test,
                         prediction_1D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_SNR_CRC.pdf',
                         display=False,
                         title='Model 1D prediction vs SNR (including CRC)',
                         predict_mean=True,
                         zoom_SNR=True,
                         zoom_window=(0,70)
                        #  nb_intervals=8,
                         )

plot_predict_wrt_SNR_CRC(SNR_test,
                         prediction_2D,
                         true_labels=true_test,
                         save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_SNR_CRC.pdf',
                         display=False,
                         title='Model 2D prediction vs SNR (including CRC)',
                         predict_mean=True,
                         zoom_SNR=True,
                         zoom_window=(0,70)
                        #  nb_intervals=8,
                         )

# plot_predict_wrt_SNR_CRC(sigma_test,
#                          prediction_1D,
#                          true_labels=true_test,
#                          save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_wrt_sigma_CRC',
#                          display=False,
#                          title='Model 1D Predictions vs Bckg level (including CRC)',
#                          predict_mean=True,
#                          nb_intervals=8,
#                          x_label='Background noise level (std dev)')

# plot_predict_wrt_SNR_CRC(sigma_test,
#                          prediction_2D,
#                          true_labels=true_test,
#                          save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_wrt_sigma_CRC',
#                          display=False,
#                          title='Model 2D Predictions vs Bckg level (including CRC)',
#                          predict_mean=True,
#                          nb_intervals=8,
#                          x_label='Background noise level (std dev)')

# ------------ Per event -------------------

prediction_1D_event = np.append(prediction_1D_event, pred_CRC_1D_event)
prediction_2D_event = np.append(prediction_2D_event, pred_CRC_2D_event)

true_test_event = np.append(true_test_event, np.zeros(len(pred_CRC_1D_event)) + 2) # label 2 for CRC
print("--------- Event predictions ----------------")
# print(pred_CRC_1D_event)
print("Mean prediction CRC 1D for events: ", np.mean(pred_CRC_1D_event))
print("Standard deviation prediction CRC 1D for events: ", np.std(pred_CRC_1D_event))

print("Mean prediction CRC 2D for events: ", np.mean(pred_CRC_2D_event))
print("Standard deviation prediction CRC 2D for events: ", np.std(pred_CRC_2D_event))


plot_predict_wrt_SNR_CRC(SNR_event,
                        prediction_1D_event,
                        true_labels=true_test_event,
                        save_path=f'{master_path}/plots/{dataset_name}/model1D_prediction_event_wrt_SNR_CRC.pdf',
                        display=False,
                        title='Model 1D prediction for full events vs SNR (including CRC)',
                        predict_mean=True)

plot_predict_wrt_SNR_CRC(SNR_event,
                        prediction_2D_event,
                        true_labels=true_test_event,
                        save_path=f'{master_path}/plots/{dataset_name}/model2D_prediction_event_wrt_SNR_CRC.pdf',
                        display=False,
                        title='Model 2D prediction for full events vs SNR (including CRC)',
                        predict_mean=True)

mask_CRC_question = (pred_CRC_2D_event <= .9)

coord_CRC_question = np.int64(np.append(file_index_CRC[mask_CRC_question].reshape(-1,1), event_index_CRC[mask_CRC_question].reshape(-1,1), axis=1))
# print(coord_CRC_question, len(coord_CRC_question[:,0]))

# print(np.unique(coord_CRC_question, axis=0, return_counts=True))
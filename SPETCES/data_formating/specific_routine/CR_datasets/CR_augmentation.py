"""
Module for data augmentation routines specific to CR datasets in SPETCES.
"""


import numpy as np
from glob import glob
import grand.dataio.root_trees as rt
import os
import matplotlib.pyplot as plt

from SPETCES.data_formating.traces_gathering import number_of_antennas_in_event_noise, measure_SNR_one_channel_1024, measure_SNR_one_channel_512
from SPETCES.imported_fcts import master_path, fs, four_notch_filters


def augment_CR_dataset(dataset_path, 
                       output_path, 
                       augmentation_factor=3,
                       trace_length=384,
                       both_1024_512=True,
                       proportion_train=0.8
                       ):
    """
    Augment the CR dataset by randomly cropping the traces so that the signal is contained in the new array
    Inputs:
        - dataset_path: path to the txt file containing all events selected for the dataset
        - output_path: path to save the augmented dataset
        - augmentation_factor: number of augmented samples to generate per original sample
        - trace_length: length of the cropped traces
    Outputs:
        - Augmented dataset saved in output_path
    """
    # Load the dataset
    file_names_CRC = np.loadtxt(dataset_path, dtype='str')

    traces_to_augment = np.zeros((0, 1024, 2), dtype=int)
    info_to_augment = np.zeros((0, 11))

    for i in range(len(file_names_CRC[:,0])) :
        fname = file_names_CRC[i,0]
        event_id = int(file_names_CRC[i,1])

        time_ref = fname.split('_')[1] + '_' + fname.split('_')[2]
        year = time_ref[0:4]
        month = time_ref[4:6]
        file_number = np.int64(fname.split('_')[1] + fname.split('_')[2])
        file_path = glob(f'/sps/grand/data/gp80/GrandRoot/{year}/{month}/{fname}')[0]

        # Opening the root file
        file_root = rt.DataFile(file_path)

        du_id = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{year}/{month}/{fname}/DU_id.txt', dtype=str)
        cut_param = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{year}/{month}/{fname}/Recons_param_table.txt')
        PWF = cut_param[np.argwhere(np.int32(du_id[:,4]) == event_id).reshape(-1)][0,2:4]

        indices_to_keep = du_id[du_id[:,4].astype(int) == event_id][:,5].astype(int)

        file_root.tadc.get_entry(event_id)

        for j in range(len(indices_to_keep)) :
            index = indices_to_keep[j]
            trace = np.zeros((1024, 2), dtype=int)
            try :
                trace[:,0] = file_root.tadc.trace_ch[index][1]
                trace[:,1] = file_root.tadc.trace_ch[index][2]
                # trace[2] = file_root.tadc.trace_ch[index][3]
                # continue
                info = np.zeros((1, 11))
                info[0, 0] = j
                info[0, 1] = file_number
                info[0, 2] = event_id
                info[0, 3] = len(indices_to_keep)
                info[0, 4] = file_root.tadc.du_id[index]
                info[0, 5] = measure_SNR_one_channel_1024(trace[:,0])
                info[0, 6] = measure_SNR_one_channel_1024(trace[:,1])
                info[0, 7] = np.std(trace[512:, 0])
                info[0, 8] = np.std(trace[512:, 1])
                info[0, 9] = PWF[1]
                info[0, 10] = PWF[0]

                trace[:,0] = trace[:,0] - np.mean(trace[512:,0])
                trace[:,1] = trace[:,1] - np.mean(trace[512:,1])

            except :
                if both_1024_512 :
                    trace_512 = np.zeros((512, 2), dtype=int)
                    trace_512[:,0] = file_root.tadc.trace_ch[index][1]
                    trace_512[:,1] = file_root.tadc.trace_ch[index][2]
                    # trace[2] = file_root.tadc.trace_ch[index][3][:512]
                    # continue
                    info = np.zeros((1, 11))
                    info[0, 0] = j
                    info[0, 1] = file_number
                    info[0, 2] = event_id
                    info[0, 3] = len(indices_to_keep)
                    info[0, 4] = file_root.tadc.du_id[index]
                    info[0, 5] = measure_SNR_one_channel_512(trace_512[:,0])
                    info[0, 6] = measure_SNR_one_channel_512(trace_512[:,1])
                    info[0, 7] = np.std(trace_512[300:, 0])
                    info[0, 8] = np.std(trace_512[300:, 1])
                    info[0, 9] = PWF[1]
                    info[0, 10] = PWF[0]

                    trace_512[:,0] = trace_512[:,0] - np.mean(trace_512[:,0])
                    trace_512[:,1] = trace_512[:,1] - np.mean(trace_512[:,1])

                    trace = np.append(trace_512, np.zeros((512,2), dtype=int), axis=0)
                else:
                    continue
            
            if (info[j,7] > 10) & (info[j,7] < 20) & (info[j,8] > 10) & (info[j,8] < 20) & (np.array([info[j,5] >=4, info[j,6] >=4]).any(axis=0)):
                info_to_augment = np.append(info_to_augment, info, axis=0)
                traces_to_augment = np.append(traces_to_augment, [trace], axis=0)


    nb_data = traces_to_augment.shape[0]
    nb_train = int(nb_data * proportion_train * 0.9) # training is 90% of the training+validation set
    nb_validation = int(nb_data * proportion_train * 0.1) # validation is 10% of the training+validation set
    nb_test = nb_data - nb_train - nb_validation

    traces_train = traces_to_augment[:nb_train]
    traces_validation = traces_to_augment[nb_train:nb_train+nb_validation]
    traces_test = traces_to_augment[nb_train+nb_validation:]

    # Now we fill these arrays with the augmented data
    augmented_train_trace = np.zeros((augmentation_factor * nb_train, trace_length, 2), dtype=int)
    augmented_validation_trace = np.zeros((augmentation_factor * nb_validation, trace_length, 2), dtype=int)
    augmented_test_trace = np.zeros((augmentation_factor * nb_test, trace_length, 2), dtype=int)

    augmented_train_info = np.zeros((augmentation_factor * nb_train, 11))
    augmented_validation_info = np.zeros((augmentation_factor * nb_validation, 11))
    augmented_test_info = np.zeros((augmentation_factor * nb_test, 11))

    event_id_transform = np.power(10, np.int32(np.log10(augmentation_factor)))

    for k in range(augmentation_factor) :
        for l in range(nb_train) :
            n_max = np.argmax(np.sqrt(traces_train[l,:,0]**2 + traces_train[l,:,1]**2)) - 30
            n = np.random.randint(0, n_max)
            
            augmented_train_trace[k*nb_train + l] = traces_train[l, n:n+trace_length, :2]

            info_temp = info_to_augment[l].copy()
            info_temp[2] = info_temp[2]*event_id_transform + k # we change the event id to be able to identify the augmented samples
            augmented_train_info[k*nb_train + l] = info_temp

        for m in range(nb_validation) :
            n_max = np.argmax(np.sqrt(traces_validation[m,:,0]**2 + traces_validation[m,:,1]**2)) - 30
            n = np.random.randint(0, n_max)

            augmented_validation_trace[k*nb_validation + m] = traces_validation[m, n:n+trace_length, :2]

            info_temp = info_to_augment[nb_train + m].copy()
            info_temp[2] = info_temp[2]*event_id_transform + k # we change the event id to be able to identify the augmented samples
            augmented_validation_info[k*nb_validation + m] = info_temp

        for n in range(nb_test) :
            n_max = np.argmax(np.sqrt(traces_test[n,:,0]**2 + traces_test[n,:,1]**2)) - 30
            n = np.random.randint(0, n_max)

            augmented_test_trace[k*nb_test + n] = traces_test[n, n:n+trace_length, :2]

            info_temp = info_to_augment[nb_train + nb_validation + n].copy()
            info_temp[2] = info_temp[2]*event_id_transform + k # we change the event id to be able to identify the augmented samples
            augmented_test_info[k*nb_test + n] = info_temp
    
    # We save the augmented dataset

    np.save(f'{output_path}/CR_test_dataset_{nb_test}_augment_{augmentation_factor}_traces_noise15_SNR4_traces.npy', augmented_test_trace)
    np.save(f'{output_path}/CR_test_dataset_{nb_test}_augment_{augmentation_factor}_traces_noise15_SNR4_info.npy', augmented_test_info)

    np.save(f'{output_path}/CR_train_dataset_{nb_train}_augment_{augmentation_factor}_traces_noise15_SNR4_traces.npy', augmented_train_trace)
    np.save(f'{output_path}/CR_train_dataset_{nb_train}_augment_{augmentation_factor}_traces_noise15_SNR4_info.npy', augmented_train_info)

    np.save(f'{output_path}/CR_validation_dataset_{nb_validation}_augment_{augmentation_factor}_traces_noise15_SNR4_traces.npy', augmented_validation_trace)
    np.save(f'{output_path}/CR_validation_dataset_{nb_validation}_augment_{augmentation_factor}_traces_noise15_SNR4_info.npy', augmented_validation_info)

if __name__ == "__main__":

    dataset_name = 'heavymodel_realCR_6_384_1024trace'


    list_CRC_path = f'/pbs/home/j/jlavoisier/proceeding_data/candidates/marion_202512/candidates_marion_202512_sure_list.txt'

    output_path = f'/sps/grand/jlavoisier/output/ML_cuts/datasets/{dataset_name}/train_test_datasets'

    augment_CR_dataset(dataset_path=list_CRC_path,
                       output_path=output_path,
                       augmentation_factor=5,
                       trace_length=384,
                       both_1024_512=False,
                       proportion_train=0.8
                       )
"""

This code creates data files containing traces used to validate the model's ability to classify signals as such after training.

"""

"""

This code creates one .npy file per file and event in input of the python file.

"""

import numpy as np
from glob import glob
import grand.dataio.root_trees as rt
import os
import matplotlib.pyplot as plt

from SPETCES.data_formating.traces_gathering import number_of_antennas_in_event_noise, measure_SNR_one_channel_1024, measure_SNR_one_channel_512
from SPETCES.imported_fcts import master_path, fs, four_notch_filters

def accumulate_traces(file_path,
                      event_index,
                      number_of_points_in_traces=512,
                      number_augmentation = 0
                      ) :
    """
    Entries:
        file_path : str, path the root file containing the events to be used for the dataset
        event_index : int, event to take the tarces from
    Output:
        a new array of traces to be saved in the dataset, bigger than traces_to_save
    """

    file_root = rt.DataFile(file_path)
    traces_to_save = np.zeros((0, number_of_points_in_traces, 2), dtype=int)
    info_to_save = np.zeros((0, 11))

    fname = file_path.split('/')[-1]
    fyear = fname.split('_')[1][0:4]
    fmonth = fname.split('_')[1][4:6]
    file_number = np.int64(fname.split('_')[1] + fname.split('_')[2])

    du_id = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{fyear}/{fmonth}/{fname}/DU_id.txt', dtype=str)
    cut_param = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{fyear}/{fmonth}/{fname}/Recons_param_table.txt')

    try:
        PWF = cut_param[np.argwhere(np.int32(du_id[:,4]) == event_index).reshape(-1)][0,2:4]
    except:
        print(f"Event {event_index} of file {fname} has a problem, skipping it.")
        return None

    nb_ant, indices_to_keep = number_of_antennas_in_event_noise(file_root, event_index)
    
    if nb_ant == 0 :
        return None
    
    file_root.tadc.get_entry(event_index)

    for j in range(nb_ant) :
        index = int(indices_to_keep[j])
        # trace = np.zeros((3, 1024), dtype=int)
        # if traces are 1024 points long
        try :
            trace = np.zeros((1024, 2), dtype=int)
            trace[:,0] = file_root.tadc.trace_ch[index][1]
            trace[:,1] = file_root.tadc.trace_ch[index][2]
            # trace[2] = file_root.tadc.trace_ch[index][3]
            # continue
            info = np.zeros((1, 11))
            info[0, 0] = j
            info[0, 1] = file_number
            info[0, 2] = event_index*10 + number_augmentation
            info[0, 3] = nb_ant
            info[0, 4] = file_root.tadc.du_id[index]
            info[0, 5] = measure_SNR_one_channel_1024(trace[:,0])
            info[0, 6] = measure_SNR_one_channel_1024(trace[:,1])
            info[0, 7] = np.std(trace[512:, 0])
            info[0, 8] = np.std(trace[512:, 1])
            info[0, 9] = PWF[1]
            info[0, 10] = PWF[0]

            trace[:,0] = trace[:,0] - np.mean(trace[:,0])
            trace[:,1] = trace[:,1] - np.mean(trace[:,1])

            n_max = np.argmax(np.sqrt(trace[:,0]**2 + trace[:,1]**2)) - 30
            # n_min = np.argmax(np.sqrt(trace[:,0]**2 + trace[:,1]**2)) + 40 - 512
            n_min = 0
            n = np.random.randint(n_min, n_max)
            # print("n_min = ", n_min, " n_max = ", n_max)

            # n=0
            traces = trace[n:n+number_of_points_in_traces,:2]
            # traces = trace[:512,:2]

            # # ---------------------
            # # MAX DATASETS
            # if np.max(trace[:,0]) > np.max(trace[:,1]):
            #     ch_max = 0
            # else:
            #     ch_max = 1

            # pos_max = np.argmax(trace[:,ch_max])
            # traces = trace[pos_max-50:pos_max+50,:2]


            # if n>=0:
            #     traces = trace[n:n+512,:2]
            # else:
            #     traces = np.append(trace[n:,:2], trace[:n+512,:2], axis=0)

            info_to_save = np.append(info_to_save, info, axis=0)
            traces_to_save = np.append(traces_to_save, [traces], axis=0)

        
        # if traces are 512 points long (can happen from october 2025)
        except :
            continue
            print('traces are 512 points long.')
            trace = np.zeros((512, 2), dtype=int)
            trace[:,0] = file_root.tadc.trace_ch[index][1]
            trace[:,1] = file_root.tadc.trace_ch[index][2]
            # trace[2] = file_root.tadc.trace_ch[index][3]
            info = np.zeros((1, 11))
            info[0, 0] = j
            info[0, 1] = file_number
            info[0, 2] = event_index
            info[0, 3] = nb_ant
            info[0, 4] = file_root.tadc.du_id[index]
            info[0, 5] = measure_SNR_one_channel_512(trace[:,0])
            info[0, 6] = measure_SNR_one_channel_512(trace[:,1])
            info[0, 7] = np.std(trace[300:, 0])
            info[0, 8] = np.std(trace[300:, 1])
            info[0, 9] = PWF[1]
            info[0, 10] = PWF[0]

            n_max = np.argmax(np.sqrt(trace[:,0]**2 + trace[:,1]**2)) - 30
            # n_min = np.argmax(np.sqrt(trace[:,0]**2 + trace[:,1]**2)) + 40 - 512
            n_min = 0
            n = np.random.randint(n_min, n_max)
            print("n_min = ", n_min, " n_max = ", n_max)

            # n=0
            traces = trace[n:n+number_of_points_in_traces,:2]
            # traces = trace[:512,:2]

            # # ---------------------
            # # MAX DATASETS

            # if np.max(trace[:,0]) > np.max(trace[:,1]):
            #     ch_max = 0
            # else:
            #     ch_max = 1

            # pos_max = np.argmax(trace[:,ch_max])
            # print("pos of max before correction: ", pos_max)
            # if pos_max < 50:
            #     pos_max = 50
            # traces = trace[pos_max-50:pos_max+50,:2]
            # print("traces shape is ", traces.shape)
            # if traces.shape[0] != 128:
            #     plt.plot(np.linspace(0,1024, 512), 
            #             trace[:,0],
            #             label='X channel',
            #             color='tab:blue'
            #         )
            #     plt.plot(np.linspace(0,1024, 512), 
            #             trace[:,1],
            #             label='Y channel',
            #             color='tab:orange'
            #             )
            #     plt.savefig(f'/sps/grand/jlavoisier/output/ML_cuts/plots/00_error/example_trace_{file_number}_{event_index}_{j}.png')

            info_to_save = np.append(info_to_save, info, axis=0)
            traces_to_save = np.append(traces_to_save, [traces], axis=0)

    

        # # if you want to see the traces being saved
        # plt.plot(np.linspace(0,1024, 512), 
        #          traces[:,0],
        #          label='X channel',
        #          color='tab:blue'
        #      )
        # plt.plot(np.linspace(0,1024, 512), 
        #          traces[:,1],
        #         label='Y channel',
        #         color='tab:orange'
        #         )
        # plt.title(f'Example trace from {file_number}')
        # plt.xlabel('Time [ns]', fontsize=16)
        # plt.ylabel('ADC counts', fontsize=16)
        # plt.legend(fontsize=12)
        # plt.savefig(f'{master_path}/datasets/dataset_verif/{list_candidates}/trace/example_trace_{file_number}_{event_index}_{j}.png')
        # plt.close()

    
    return(traces_to_save, info_to_save)

if __name__ == '__main__' :
    list_candidates = 'marion_202512_sure'

    number_of_points_in_traces = 384
    number_augmentation = 4

    file_names_ICRC2025 = np.loadtxt(f'/pbs/home/j/jlavoisier/proceeding_data/candidates/marion_202512/candidates_{list_candidates}.txt', dtype='str')
    

    os.makedirs(f'{master_path}/datasets/dataset_verif/{list_candidates}/trace', exist_ok=True)

    # Data augmentation
    traces_to_save_all = np.zeros((0, number_of_points_in_traces, 2), dtype=int)
    info_to_save_all = np.zeros((0, 11))


    for i in range(len(file_names_ICRC2025[:,0])) :
        if file_names_ICRC2025[i,3] == '0' :
            continue
        file_name = file_names_ICRC2025[i,0]
        time_ref = file_name.split('_')[1] + '_' + file_name.split('_')[2]
        year = time_ref[0:4]
        month = time_ref[4:6]
        file_path = glob(f'/sps/grand/data/gp80/GrandRoot/{year}/{month}/{file_name}')[0]
        # print(f"Processing file: {file_name}")

        event_index = np.int64(file_names_ICRC2025[i,2])

        try:
            traces, info = accumulate_traces(file_path=file_path,
                                event_index=event_index,
                                number_of_points_in_traces=number_of_points_in_traces,
                                number_augmentation=number_augmentation
                                )
            
        except:
            print(f"Event {event_index} of file {file_name} has a problem, skipping it.")
            continue
        fname = file_path.split('/')[-1]
        

        # np.save(f'{master_path}/datasets/dataset_verif/{list_candidates}/{time_ref}_{event_index}_traces.npy',
        #         traces)
        # np.save(f'{master_path}/datasets/dataset_verif/{list_candidates}/{time_ref}_{event_index}_info.npy',
        #         info)
        print(f"Event {event_index} of file {time_ref} treated.")

        for j in range(info.shape[0]) :
            if (info[j,7] > 10) & (info[j,7] < 20) & (info[j,8] > 10) & (info[j,8] < 20) & (np.array([info[j,5] >=4, info[j,6] >=4]).any(axis=0)) :
                traces_to_save_all =  np.append(traces_to_save_all, [traces[j]], axis=0)
                info_to_save_all =  np.append(info_to_save_all, [info[j]], axis=0)
        

        number_CRCtraces = traces_to_save_all.shape[0]

        liste_signal = np.arange(number_CRCtraces)
        np.random.shuffle(liste_signal)

        number_CRCtrain = int(number_CRCtraces*0.72)
        number_CRCvalidation = int(number_CRCtraces*0.08)
        number_CRCtest = number_CRCtraces - number_CRCtrain - number_CRCvalidation

    dataset_name = 'heavymodel_realCR_6_384_1024trace'


    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/test_{list_candidates}_{number_CRCtest}_{number_augmentation}_traces.npy',
                traces_to_save_all[number_CRCtrain + number_CRCvalidation:])
    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/test_{list_candidates}_{number_CRCtest}_{number_augmentation}_info.npy',
                info_to_save_all[number_CRCtrain + number_CRCvalidation:])
    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/train_{list_candidates}_{number_CRCtrain}_{number_augmentation}_traces.npy',
                traces_to_save_all[:number_CRCtrain])
    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/train_{list_candidates}_{number_CRCtrain}_{number_augmentation}_info.npy',
                info_to_save_all[:number_CRCtrain])
    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/validation_{list_candidates}_{number_CRCvalidation}_{number_augmentation}_traces.npy',
                traces_to_save_all[number_CRCtrain:number_CRCtrain + number_CRCvalidation])
    np.save(f'{master_path}/datasets/{dataset_name}/raw_dataset/validation_{list_candidates}_{number_CRCvalidation}_{number_augmentation}_info.npy',
                info_to_save_all[number_CRCtrain:number_CRCtrain + number_CRCvalidation])
    # np.save(f'/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_verif/marion_202512_sure/traces.npy',
    #             traces_to_save_all)
    # np.save(f'/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_verif/marion_202512_sure/info.npy',
    #             info_to_save_all)




    # file_path = sys.argv[1]
    # event_index = int(sys.argv[2])

    # traces = accumulate_traces(file_path=file_path,
    #                            event_index=event_index)
    
    # fname = file_path.split('/')[-1]
    
    # np.save('/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_verif/ICRC2025/' + fname.replace('.root', '.npy'),
    #         traces)
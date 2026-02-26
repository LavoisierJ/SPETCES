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

from SPETCES.data_formating.traces_gathering import number_of_antennas_in_event_noise, measure_SNR_one_channel_1024
from SPETCES.imported_fcts import master_path, four_notch_filters

def accumulate_traces(file_path,
                      event_index
                      ) :
    """
    Entries:
        file_path : str, path the root file containing the events to be used for the dataset
        event_index : int, event to take the tarces from
    Output:
        a new array of traces to be saved in the dataset, bigger than traces_to_save
    """

    file_root = rt.DataFile(file_path)
    traces_to_save = np.zeros((0, 512, 2), dtype=int)
    info_to_save = np.zeros((0, 11))

    file_name = file_path.split('/')[-1]
    fname = file_name.split('_')[1] + '_' + file_name.split('_')[2]
    fyear = fname.split('_')[0][0:4]
    fmonth = fname.split('_')[0][4:6]
    file_number = np.int64(fname.split('_')[1] + fname.split('_')[2])

    du_id = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{fyear}/{fmonth}/{fname}/DU_id.txt', dtype=str)
    cut_param = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{fyear}/{fmonth}/{fname}/Recons_param_table.txt')

    PWF = cut_param[np.argwhere(np.int32(du_id[:,4]) == event_index).reshape(-1)][0,2:4]

    nb_ant, indices_to_keep = number_of_antennas_in_event_noise(file_root, event_index)
    
    if nb_ant == 0 :
        return None
    
    file_root.tadc.get_entry(event_index)

    for j in range(nb_ant) :
        index = int(indices_to_keep[j])
        # trace = np.zeros((3, 1024), dtype=int)
        trace = np.zeros((1024, 2), dtype=int)
        trace[:,0] = file_root.tadc.trace_ch[index][1]
        trace[:,1] = file_root.tadc.trace_ch[index][2]
        # trace[2] = file_root.tadc.trace_ch[index][3]
        info = np.zeros((1, 11))
        info[0, 0] = j
        info[0, 1] = file_number
        info[0, 2] = event_index
        info[0, 3] = nb_ant
        info[0, 4] = file_root.tadc.du_id[index]
        info[0, 5] = measure_SNR_one_channel_1024(trace[:,0])
        info[0, 6] = measure_SNR_one_channel_1024(trace[:,1])
        info[0, 7] = np.std(trace[512:, 0])
        info[0, 8] = np.std(trace[512:, 1])
        info[0, 9] = PWF[1]
        info[0, 10] = PWF[0]

        info_to_save = np.append(info_to_save, info, axis=0)

        trace[:,0] = trace[:,0] - np.mean(trace[:,0])
        trace[:,1] = trace[:,1] - np.mean(trace[:,1])

        # n = np.random.randint(0, 512)
        # while (np.max(trace[n+20:n+512,0])!= np.max(trace[:,0])) and (np.max(trace[n+20:n+512,1])!= np.max(trace[:,1])):
        #     n = np.random.randint(0, 512)
        n=0
        
        # # if you want to see the traces being saved
        # plt.plot(np.linspace(0,1024, 512), 
        #          trace[n:n+512,0],
        #          label='X channel',
        #          color='tab:blue'
        #      )
        # plt.plot(np.linspace(0,1024, 512), 
        #          trace[n:n+512,1],
        #         label='Y channel',
        #         color='tab:orange'
        #         )
        # plt.title(f'Example trace from {file_number} after filtering')
        # plt.xlabel('Time [ns]', fontsize=16)
        # plt.ylabel('ADC counts', fontsize=16)
        # plt.legend(fontsize=14)
        # plt.savefig(f'{master_path}/datasets/dataset_verif/{list_candidates}/trace/example_trace_{file_number}_{event_index}_{j}.png')
        # plt.close()

        traces_to_save = np.append(traces_to_save, [trace[n:n+512,:]], axis=0)
    
    return(traces_to_save, info_to_save)

if __name__ == '__main__' :
    list_candidates = '2025/07'

    file_names_CRC = np.loadtxt(f'/sps/grand/jlavoisier/output/data_treatment/{list_candidates}/Flagged_events.txt', dtype='str')

    os.makedirs(f'{master_path}/datasets/dataset_flagged/{list_candidates}/', exist_ok=True)
    for i in range(len(file_names_CRC[:,0])) :
        time_ref = file_names_CRC[i,0]
        year = time_ref.split('_')[1][0:4]
        month = time_ref.split('_')[1][4:6]
        file_path = glob(f'/sps/grand/data/gp80/GrandRoot/{year}/{month}/*{time_ref}*')[0]

        event_index = np.int32(file_names_CRC[i,1])

        traces, info = accumulate_traces(file_path=file_path,
                               event_index=event_index)
        
        fname = file_path.split('/')[-1]
        

        np.save(f'{master_path}/datasets/dataset_flagged/{list_candidates}/{time_ref}_{event_index}_traces.npy',
                traces)
        np.save(f'{master_path}/datasets/dataset_flagged/{list_candidates}/{time_ref}_{event_index}_info.npy',
                info)
        print(f"Event {event_index} of file {time_ref} treated.")
        




    # file_path = sys.argv[1]
    # event_index = int(sys.argv[2])

    # traces = accumulate_traces(file_path=file_path,
    #                            event_index=event_index)
    
    # fname = file_path.split('/')[-1]
    
    # np.save('/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_verif/ICRC2025/' + fname.replace('.root', '.npy'),
    #         traces)
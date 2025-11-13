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

from SPETCES.data_formating.extract_traces import number_of_antennas_in_event_noise
from SPETCES.imported_fcts import master_path

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

        n = np.random.randint(0, 512)
        while (np.max(trace[n+20:n+512,0])!= np.max(trace[:,0])) and (np.max(trace[n+20:n+512,1])!= np.max(trace[:,1])):
            n = np.random.randint(0, 512)

        traces_to_save = np.append(traces_to_save, [trace[n:n+512,:]], axis=0)
    
    return(traces_to_save)

if __name__ == '__main__' :
    list_candidates = 'pengxiong_202510'

    file_names_ICRC2025 = np.loadtxt(f'/pbs/home/j/jlavoisier/proceeding_data/candidates/candidates_{list_candidates}.txt', dtype='str')

    os.makedirs(f'{master_path}/datasets/dataset_verif/{list_candidates}/', exist_ok=True)
    for i in range(len(file_names_ICRC2025[:,0])) :
        time_ref = file_names_ICRC2025[i,0]
        year = time_ref[0:4]
        month = time_ref[4:6]
        file_path = glob(f'/sps/grand/data/gp80/GrandRoot/{year}/{month}/*{time_ref}*')[0]

        event_index = np.int32(file_names_ICRC2025[i,1])

        traces = accumulate_traces(file_path=file_path,
                               event_index=event_index)
        
        fname = file_path.split('/')[-1]
        

        np.save(f'{master_path}/datasets/dataset_verif/{list_candidates}/{time_ref}_{event_index}.npy',
                traces)
        print(f"Event {event_index} of file {time_ref} treated.")
        




    # file_path = sys.argv[1]
    # event_index = int(sys.argv[2])

    # traces = accumulate_traces(file_path=file_path,
    #                            event_index=event_index)
    
    # fname = file_path.split('/')[-1]
    
    # np.save('/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_verif/ICRC2025/' + fname.replace('.root', '.npy'),
    #         traces)
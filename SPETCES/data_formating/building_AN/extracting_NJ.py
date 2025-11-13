import numpy as np
import grand.dataio.root_trees as rt
from glob import glob
import os.path
import sys

from SPETCES.imported_fcts import master_path

# --------------------- For Noise ---------------------
def number_of_antennas_in_event_noise(root_file_CD,
                                      index) :
    """
    Entries :
        root_file_CD : root file
        index : index of event
    Output :
        Number of effective antennas in the event
        List of indexes inside the event containing the effective antennas
    """

    root_file_CD.tadc.get_entry(index)

    number_of_entries = len(root_file_CD.tadc.du_id)
    memory_DUid = np.array([])

    # list containing the antennas we'll be looking through to analyze the event
    indices_to_keep = np.array([])

    for i in range(number_of_entries) :
        if np.isin(root_file_CD.tadc.du_id[i], memory_DUid) or root_file_CD.tadc.du_seconds[i] == 0 :
            # test if the antenna is duplicated or if it is empty
            continue
        else :
            memory_DUid = np.append(memory_DUid, root_file_CD.tadc.du_id[i])
            indices_to_keep = np.append(indices_to_keep, i)
    
    return(len(indices_to_keep), indices_to_keep)

# -------------------- For Signal --------------------------

def extract_trigger_parameters(trace, trigger_config, baseline=0):
    # Extract the trigger infos from a trace

    # Parameters :
    # ------------
    # trace, numpy.ndarray: 
    # traces in ADC unit
    # trigger_config, dict:
    # the trigger parameters set in DAQ

    # Returns :
    # ---------
    # Index in the trace when the first T1 crossing happens
    # Indices in the trace of T2 crossing happens
    # Number of T2 crossings
    # Q, Peak/NC

    # Find the position of the first T1 crossing
    index_t1_crossing = np.where((trace) > trigger_config["th1"],
                                 np.arange(len(trace)), -1)
    dict_trigger_infos = dict()
    mask_T1_crossing = (index_t1_crossing != -1)
    if sum(mask_T1_crossing) == 0:
        # No T1 crossing 
        raise ValueError("No T1 crossing!")
    dict_trigger_infos['index_T1_crossing'] = None
    # Tquiet to decide the quiet time before the T1 crossing 
    for i in index_t1_crossing[mask_T1_crossing]:
       # Abs value not exceeds the T1 threshold
        if i - trigger_config["t_quiet"]//2 < 0:
            raise ValueError("Not enough data before T1 crossing!")
        if np.all((trace[np.max([0, i - trigger_config['t_quiet'] // 2]):i]) <= trigger_config["th1"]):
            dict_trigger_infos["index_T1_crossing"] = i
            # the first T1 crossing satisfying the quiet condition
            break
    if dict_trigger_infos['index_T1_crossing'] == None:
        raise ValueError("No T1 crossing with Tquiet satified!")
    # The trigger logic works for the timewindow given by T_period after T1 crossing.
    # Count number of T2 crossings, relevant pars: T2, NCmin, NCmax, T_sepmax
    # From ns to index, divided by two for 500MHz sampling rate
    
    period_after_T1_crossing = trace[dict_trigger_infos["index_T1_crossing"]:dict_trigger_infos["index_T1_crossing"]+trigger_config['t_period']//2]
    # All the points above +T2
    positive_T2_crossing = (np.array(period_after_T1_crossing) > trigger_config['th2']).astype(int)
    # Positive crossing, the point before which is below T2.
    mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1

    # Register the first T1 crossing as a T2 crossing
    mask_first_T1_crossing = np.zeros(len(period_after_T1_crossing), dtype=bool)
    mask_first_T1_crossing[0] = True

    mask_first_T1_crossing[1:] = (mask_T2_crossing_positive)
    index_T2_crossing = np.arange(len(period_after_T1_crossing))[mask_first_T1_crossing]
    n_T2_crossing = 1 # Starting from the first T1 crossing.
    dict_trigger_infos["index_T2_crossing"] = [0]
    if len(index_T2_crossing) > 1:
        for i, j in zip(index_T2_crossing[:-1], index_T2_crossing[1:]):
            # The separation between successive T2 crossings
            time_separation = (j - i) * 2
            if time_separation < trigger_config["t_sepmax"]:
                n_T2_crossing += 1
                dict_trigger_infos["index_T2_crossing"].append(j)
            else:
                # Violate the maximum separation, fail to trigger
                raise ValueError(f"Violating Tsepmax, the separation is {time_separation} ns.")
    else:
        n_T2_crossing = 1
        j = 1
    # Change the reference of indices of T2 crossing
    dict_trigger_infos["index_T2_crossing"] = np.array(dict_trigger_infos["index_T2_crossing"]) + dict_trigger_infos["index_T1_crossing"]
    dict_trigger_infos["NC"] = n_T2_crossing
    
    # Calulate the peak value
    dict_trigger_infos["Q"] = (np.max(np.abs(period_after_T1_crossing[:j])) - baseline) / dict_trigger_infos["NC"]
    return dict_trigger_infos

dict_trigger_parameter = dict([
  ("t_quiet", 512),
  ("t_period", 512),
  # ("t_sepmax", 20),
  ("t_sepmax", 50),
  ("nc_min", 2),
  ("nc_max", 7),
  ("q_min", 0),
  ("q_max", 255),
  # ("th1", 100),
  # ("th2", 50),
  ("th1",70),
  ("th2", 60),
  # Configs of readout timewindow
  ("t_pretrig", 960),
  ("t_overlap", 64),
  ("t_posttrig", 1024)
  ])

def pass_T1(list_traces):
    """
    Inputs
        list_traces: list of list, containing the traces of the 3 channels for an antenna (X,Y,Z), shape(3,1024)
    Output
        Boolean indicating whether the signal passes the trigger T1 (trigger tested over channels X and Y)
    """
    for v in range(2) :
        trace = list_traces[v]
        try:
            trigger_infos = extract_trigger_parameters(trace, dict_trigger_parameter)
            if trigger_infos["NC"] >= dict_trigger_parameter["nc_min"] and trigger_infos["NC"] <= dict_trigger_parameter["nc_max"]:
                # triggered_ant.append(i)
                indicator = True
                break
            else:
                indicator = False
        except ValueError as e:
            # No T1 crossing, no trigger
            # print(k, ": No trigger.")
            indicator = False
            pass
    return(indicator)


def number_of_antennas_in_event_signal(root_file_DC2,
                                       index) :
    """
    Entries :
        root_file_DC2 : root file
        index : index of event
    Output :
        Number of effective antennas in the event
        List of indexes inside the event containing the effective antennas
    """

    root_file_DC2.tadc.get_entry(index)

    number_of_entries = len(root_file_DC2.tadc.du_id)

    # list containing the antennas we'll be looking through to analyze the event
    indices_to_keep = np.array([])

    trace = np.zeros((3, 1024), dtype=int)

    for i in range(number_of_entries) :
        trace[0] = root_file_DC2.tadc.trace_ch[i][0]
        trace[1] = root_file_DC2.tadc.trace_ch[i][1]
        trace[2] = root_file_DC2.tadc.trace_ch[i][2]
        if pass_T1(trace):
            indices_to_keep = np.append(indices_to_keep, i)            
    
    return(len(indices_to_keep), indices_to_keep)


# -------------------- Create Dataset --------------------------

def accumulate_traces(file_path,
                     traces_to_save
                     ) :
    """
    Entries:
        file_path : str, path the root file containing the events to be used for the dataset
        traces_to_save : array of array, shape (-1, 3, 1024), traces to be saved in the dataset
    Output:
        a new array of traces to be saved in the dataset, bigger than traces_to_save
    """

    file_root = rt.DataFile(file_path)

    n_event = file_root.tadc.get_number_of_entries()

    for i in range(n_event) :
        nb_ant, indices_to_keep = number_of_antennas_in_event_signal(file_root, i)
        
        if nb_ant == 0 :
            continue
        
        file_root.tadc.get_entry(i)

        for j in range(nb_ant) :
            index = int(indices_to_keep[j])
            trace = np.zeros((3, 1024), dtype=int)
            trace[0] = file_root.tadc.trace_ch[index][0]
            trace[1] = file_root.tadc.trace_ch[index][1]
            trace[2] = file_root.tadc.trace_ch[index][2]

            traces_to_save = np.append(traces_to_save, [trace], axis=0)
    
    return(traces_to_save)



def create_dataset(repert_list,
                   save_path
                   ) :
    """
    Entries:
        repert_list : list of str, paths to the root files containing the events to be used for the dataset
        save_path : str, path to the output dataset
    """

    traces_to_save = np.zeros((0, 3, 1024), dtype=int)
    for file_path in repert_list :
        print(file_path)
        traces_to_save = accumulate_traces(file_path,
                                          traces_to_save
                                          )
        print(traces_to_save.shape)
    
    np.save(save_path, traces_to_save)
    print(f"Dataset saved at {save_path}, shape: {traces_to_save.shape}")
    return(None)




# # # -------------------- For Signal --------------------------

# repert_signal = sorted(glob('/sps/grand/DC2.1rc4/GP*ZHAireS-AN/*/adc*_L1_0000.root'))


# save_path_signal = '/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_signal_DC2rc4_AN.npy'

# create_dataset(repert_signal,
#                save_path_signal,
#                Signal_or_Noise=0
#                )

# ---------------------- For tests -----------------------

# repert_signal = [glob('/sps/grand/DC2.1rc4/GP289ZHAireS-AN/*/adc_*_L1_0000.root')[0]]
# print(repert_signal)

# save_path_signal = '/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_signal_test.npy'

# create_dataset(repert_signal,
#                save_path_signal,
#                Signal_or_Noise=0
#                )

# repert_noise = [glob('/sps/grand/data/gp80/GrandRoot/2025/07/*CD*.root')[0]]
# print(repert_noise)

# save_path_noise = '/sps/grand/jlavoisier/output/ML_cuts/datasets/dataset_noise_test.npy'

# create_dataset(repert_noise,
#                save_path_noise,
#                Signal_or_Noise=1
#                )

# ---------------------- For Job Submission ------------------------

if __name__ == "__main__":
    repert_dataset = [sys.argv[1]]
    print(np.array(repert_dataset))

    fname = repert_dataset[0].split('/')[-1]

    
    os.makedirs(f'{master_path}/datasets/dataset_building_AN/NJ_sims', exist_ok=True)
    save_path = f'{master_path}/datasets/dataset_building_AN/NJ_sims/' + fname.replace('.root', '.npy')


    create_dataset(repert_dataset,
                save_path
                )
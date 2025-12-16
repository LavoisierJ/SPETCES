import numpy as np
import grand.dataio.root_trees as rt
from glob import glob
import os.path
import sys

from SPETCES.imported_fcts import master_path, fs, four_notch_filters
from PWF_reconstruction.recons_PWF import PWF_semianalytical
from scipy.signal import hilbert

from grand import ECEF, Geodetic, GRANDCS, LTP
coord_DAQ = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)
coord_origin = coord_DAQ

# -------------------- Helper Functions ------------------------

def measure_SNR_one_channel_1024(trace
                                 ) :
    """
    Entry:
        trace : numpy array, shape (1024,), trace of one channel
    Output:
        SNR : float, Signal to Noise Ratio of the trace
    """
    # Measure the noise RMS on the last 512 samples
    noise_rms = np.std(trace[512:])
    signal_power = np.max(np.abs(trace[:512]))
    if noise_rms == 0:
        return 0
    return signal_power / noise_rms

def measure_SNR_one_channel_512(trace
                                 ) :
    """
    Entry:
        trace : numpy array, shape (512,), trace of one channel
    Output:
        SNR : float, Signal to Noise Ratio of the trace
    """
    # Measure the noise RMS on the last 256 samples
    noise_rms = np.std(trace[256:])
    signal_power = np.max(np.abs(trace[:256]))
    if noise_rms == 0:
        return 0
    return signal_power / noise_rms


def get_DU_coord(lat, long, alt, obstime, origin=coord_origin):
    # From GPS to Cartisian coordinates
    geod = Geodetic(latitude=lat, longitude=long, height=alt)
    gcs = GRANDCS(geod, obstime=obstime, location=origin)
    return gcs



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




# ---------------------- For Job Submission ------------------------

if __name__ == "__main__":
    repert_dataset = [sys.argv[1]]
    # repert_dataset = [glob('/sps/grand/data/gp80/GrandRoot/2025/07/*CD*.root')[0]]
    # repert_dataset = [glob('/sps/grand/DC2.1rc4/GP289ZHAireS-AN/*/adc_*_L1_0000.root')[0]]

    print(np.array(repert_dataset))

    Signal_or_Noise = np.int32(sys.argv[2]) # 1 for signal 0 for noise
    # Signal_or_Noise = 0
    # Signal_or_Noise = 1
    print(f"Signal_or_Noise = {Signal_or_Noise}")

    fname = repert_dataset[0].split('/')[-1]

    
    if Signal_or_Noise :
        repert_shower = '/'.join(repert_dataset[0].split('/')[:-1]) + '/shower' + fname[3:].replace('L1_0000', 'L0_0000')
        file_shower_root = rt.DataFile(repert_shower)

    traces_to_save = np.zeros((0, 3, 1024), dtype=int)
    nb_ant_for_coinc = np.array([])
    file_name = np.array([])
    root_index = np.array([])
    SNR_X = np.array([])
    SNR_Y = np.array([])
    RMS_X = np.array([])
    RMS_Y = np.array([])
    du_id_array = np.array([])
    theta_list = np.array([])
    phi_list = np.array([])

    if Signal_or_Noise:
        file_number = np.int32(fname.split('_')[1].replace('-', ''))
    else:
        file_number = np.int64(fname.split('_')[1] + fname.split('_')[2])

    file_root = rt.DataFile(repert_dataset)

    n_event = file_root.tadc.get_number_of_entries()

    event_number = 0
    for i in range(n_event) :
        if Signal_or_Noise == 1 :
            nb_ant, indices_to_keep = number_of_antennas_in_event_signal(file_root, i)
        elif Signal_or_Noise == 0 :
            nb_ant, indices_to_keep = number_of_antennas_in_event_noise(file_root, i)

        
        if nb_ant <= 4 :
            continue
        
        file_root.tadc.get_entry(i)

        lon_lat_coord = np.zeros((nb_ant, 3), dtype=float)
        x_ants = np.zeros((nb_ant, 3), dtype=float)
        t_ants = np.zeros(nb_ant, dtype=float)
        t_nanoseconds_ants = np.zeros(nb_ant, dtype=np.int64)
        t_seconds_ants = np.zeros(nb_ant, dtype=np.int64)

        for j in range(nb_ant) :
            index = int(indices_to_keep[j])
            trace = np.zeros((3, 1024), dtype=int)
            current_du_id = file_root.tadc.du_id[index]
            if Signal_or_Noise :
                trace[0] = file_root.tadc.trace_ch[index][0]
                trace[1] = file_root.tadc.trace_ch[index][1]
                trace[2] = file_root.tadc.trace_ch[index][2]
            else :
                trace[0] = four_notch_filters(file_root.tadc.trace_ch[index][1], fs)
                trace[1] = four_notch_filters(file_root.tadc.trace_ch[index][2], fs)
                trace[2] = four_notch_filters(file_root.tadc.trace_ch[index][3], fs)

                file_root.trawvoltage.get_entry(i)
                lon_lat_coord[j, 0] = file_root.trawvoltage.gps_lat[index]
                lon_lat_coord[j, 1] = file_root.trawvoltage.gps_long[index]
                lon_lat_coord[j, 2] = file_root.trawvoltage.gps_alt[index]

                gcs = get_DU_coord(lon_lat_coord[j,0],
                                   lon_lat_coord[j,1],
                                   lon_lat_coord[j,2],
                                   '1970-01-01')
                
                x_ants[j, 0] = gcs.x[0]
                x_ants[j, 1] = gcs.y[0]
                x_ants[j, 2] = gcs.z[0]

            # measure the time of detection of the pulse
            t = np.linspace(0, 2048, 1024)
            Emodulus = np.sqrt(trace[0]**2+trace[1]**2+trace[2]**2)
            hilbert_amp = np.abs(hilbert(Emodulus))
            peaktime = t[np.int32(np.argmax(hilbert_amp))]
            t_nanoseconds_ants[j] = file_root.tadc.du_nanoseconds[index] + peaktime
            t_seconds_ants[j] = file_root.tadc.du_seconds[index]

            traces_to_save = np.append(traces_to_save, [trace], axis=0)
            SNR_X = np.append(SNR_X, measure_SNR_one_channel_1024(trace[0]))
            SNR_Y = np.append(SNR_Y, measure_SNR_one_channel_1024(trace[1]))
            RMS_X = np.append(RMS_X, np.std(trace[0][512:]))
            RMS_Y = np.append(RMS_Y, np.std(trace[1][512:]))
            du_id_array = np.append(du_id_array, current_du_id)
            root_index = np.append(root_index, i)
            file_name = np.append(file_name, file_number)
        
        t_seconds_ants = t_seconds_ants - np.min(t_seconds_ants)
        t_nanoseconds_ants = t_nanoseconds_ants - t_nanoseconds_ants[np.argmin(t_seconds_ants)]
        t_ants = t_seconds_ants + t_nanoseconds_ants*1e-9

        

        if Signal_or_Noise == 1 :
            file_shower_root.tshower.get_entry(i)
            theta = file_shower_root.tshower.zenith*np.pi/180.
            phi = file_shower_root.tshower.azimuth*np.pi/180.
        else:
            theta, phi = PWF_semianalytical(x_ants, t_ants)
    
        theta_list = np.append(theta_list, np.zeros(nb_ant)+theta)
        phi_list = np.append(phi_list, np.zeros(nb_ant)+phi)

        nb_ant_for_coinc = np.append(nb_ant_for_coinc, np.zeros(nb_ant)+nb_ant)
        event_number += 1

    if traces_to_save.shape[0] == 0 :
        print("No trace selected, exiting...")
        sys.exit()
    
    if Signal_or_Noise :
        os.makedirs(f'{master_path}/gathered_traces/sims_AN/{fname}', exist_ok=True)
        save_path_traces = f'{master_path}/gathered_traces/sims_AN/{fname}/traces.npy'
        save_path_info = f'{master_path}/gathered_traces/sims_AN/{fname}/info.npy'

    else:
        os.makedirs(f'{master_path}/gathered_traces/filtered_noise/{fname}', exist_ok=True)
        save_path_traces = f'{master_path}/gathered_traces/filtered_noise/{fname}/traces.npy'
        save_path_info = f'{master_path}/gathered_traces/filtered_noise/{fname}/info.npy'

    np.save(save_path_traces, traces_to_save)
    print(f"Dataset saved at {save_path_traces}, shape: {traces_to_save.shape}")

    line_number = np.arange(len(root_index))
    info = np.array([line_number,
                     file_name,
                     root_index,
                     nb_ant_for_coinc,
                     du_id_array,
                     SNR_X,
                     SNR_Y,
                     RMS_X,
                     RMS_Y,
                     theta_list,
                     phi_list
                     ]).T
    print(info)
    np.save(save_path_info, info)
    print(f"Info saved at {save_path_info}, shape: {info.shape}")
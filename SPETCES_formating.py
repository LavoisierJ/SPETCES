"""
This code formats the GRAND data into .npy files for the SPETCES project.

It reads the data from root files using the available DU_id.txt files, extracts the traces in *_traces.npy files,
and saves relevant informations (such as SNR by channel, noise levels, as well as event indexes) in a *_info.npy file.
"""

import numpy as np
import os
import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
from glob import glob

if __name__ == "__main__":
    # File
    fname = "GP80_20250703_020024_RUN10123_CD_20dB-GP65-Y2float-62dus-CD-100000-1.root"
    # fname = "GP80_20250701_111847_RUN10122_CD_20dB-GP65-Y2float-62dus-CD-100000-1.root"
    # fname = "GP80_20250707_065719_RUN10126_CD_20dB-GP65-Y2float-10dus-TESTRUN-CD-100000-432.root" # CR candidates here

    # Path to the data
    date = fname.split('_')[1] + '_' + fname.split('_')[2]
    year = date[0:4]
    month = date[4:6]
    file_number = np.int64(fname.split('_')[1] + fname.split('_')[2])

    DU_id_path = f"/sps/grand/jlavoisier/output/data_treatment/{year}/{month}/{fname}/DU_id.txt"
    data_path = f"/sps/grand/data/gp80/GrandRoot/{year}/{month}/{fname}"

    # Read the DU_id.txt file to get the list of events and antenna indices
    DU_id = np.loadtxt(DU_id_path, dtype=str)
    event_indexes = DU_id[:, 4].astype(int)
    ant_indexes = DU_id[:, 5].astype(int)

    # Initialize lists to store the traces and info
    traces_to_save_all = np.zeros((0, 384, 2), dtype=int)
    info_to_save_all = np.zeros((0, 11))

    # Search through the root file
    root_file = rt.DataFile(data_path)
    for event_index in np.unique(event_indexes):
        # Get the corresponding antenna indices for the current event
        root_file.tadc.get_entry(event_index)  # Load the raw voltage data for the current event
        ant_indices_event = ant_indexes[event_indexes == event_index]


        # Extract the traces for the current event and antenna indices
        traces_to_save = np.zeros((len(ant_indices_event), 384, 2), dtype=int)
        info_to_save = np.zeros((len(ant_indices_event), 11))
        for i in range(len(ant_indices_event)):
            ant_index = int(ant_indices_event[i])
            trace = np.zeros((1024, 2), dtype=int)
            trace[:, 0] = root_file.tadc.trace_ch[ant_index][1]
            trace[:, 1] = root_file.tadc.trace_ch[ant_index][2]
            traces_to_save[i] = trace[:384,:]  # Keep only the first 384 samples

            # Gather the info to save for the current event and antenna index
            sigma_X = np.std(trace[512:, 0])
            sigma_Y = np.std(trace[512:, 1])
            SNR_X = np.max(trace[:512, 0]) / sigma_X
            SNR_Y = np.max(trace[:512, 1]) / sigma_Y

            info_to_save[i, 0] = 0 # To be filled with a np.arange
            info_to_save[i, 1] = file_number
            info_to_save[i, 2] = event_index
            info_to_save[i, 3] = ant_index
            info_to_save[i, 4] = root_file.tadc.du_id[ant_index]
            info_to_save[i, 5] = SNR_X
            info_to_save[i, 6] = SNR_Y
            info_to_save[i, 7] = sigma_X
            info_to_save[i, 8] = sigma_Y
            info_to_save[i, 9] = 0 # in the long term, can be filled with PWF information to make some stats
            info_to_save[i, 10] = 0

        # Append the traces and info to the lists
        traces_to_save_all = np.concatenate((traces_to_save_all, traces_to_save), axis=0)
        info_to_save_all = np.concatenate((info_to_save_all, info_to_save), axis=0)
    
    info_to_save_all[:, 0] = np.arange(info_to_save_all.shape[0])  # Fill the first column with a range of integers

    # Save the traces and info in .npy files
    output_dir = f"/sps/grand/jlavoisier/output/ML_cuts/SPETCES_cut/{year}/{month}/{fname}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/traces.npy", traces_to_save_all)
    np.save(f"{output_dir}/info.npy", info_to_save_all)


    # # Plot the traces if needed
    # os.makedirs(f"{output_dir}/traces", exist_ok=True)
    # for i in range(np.shape(traces_to_save_all)[0]):
    #     event_index = int(info_to_save_all[i, 2])
    #     ant_index = int(info_to_save_all[i, 3])
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(np.linspace(0,384*2, 384), 
    #              traces_to_save_all[i, :, 0],
    #              label='X channel',
    #              color='tab:blue'
    #             )
    #     plt.plot(np.linspace(0,384*2, 384), 
    #                 traces_to_save_all[i, :,1],
    #             label='Y channel',
    #             color='tab:orange'
    #             )
    #     plt.title(f'Example trace from {file_number}')
    #     plt.xlabel('Time [ns]', fontsize=16)
    #     plt.ylabel('ADC counts', fontsize=16)
    #     plt.legend(fontsize=12)
    #     plt.savefig(f'{output_dir}/traces/trace_{event_index}_{ant_index}.png')
    #     plt.close()
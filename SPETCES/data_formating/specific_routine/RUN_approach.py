"""
This python file builds a dataset for noise, handmade AN and test using one RUN.
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sys

from SPETCES.imported_fcts import master_path, four_notch_filters
from SPETCES.data_formating.specific_routine.building_AN.extracting_NJ import pass_T1
# The run number is given prior to this file to have both the traces.npy and info.npy files.

run_number = 'dataset_real_CR2'

divide_mean_noise = 15




# repertories = sorted(glob(f'{master_path}/gathered_traces/run/*{run_number}*'))
repertories = sorted(glob(f'{master_path}/gathered_traces/noise/*'))[70:75]
# repertories = ['/sps/grand/jlavoisier/output/ML_cuts/gathered_traces/noise/GP80_20250707_051620_RUN10126_CD_20dB-GP65-Y2float-10dus-TESTRUN-CD-100000-157.root']





save_noise_path = f'{master_path}/datasets/{run_number}/noise_dataset_{run_number}'
save_ANhm_path = f'{master_path}/datasets/{run_number}/AN_handmade_dataset_{run_number}'
save_test_path = f'{master_path}/datasets/{run_number}/test_dataset_{run_number}'
os.makedirs(f'{master_path}/datasets/{run_number}', exist_ok=True)

background_level = 15 # in ADC units
background_tolerance = 5 # in ADC units

valid_traces = np.empty((0,3,1024), dtype=np.int16)
valid_info = np.empty((0,11), dtype=np.float32)




for repert in repertories:
    # Load the traces and info files
    traces = np.load(f'{repert}/traces.npy')
    info = np.load(f'{repert}/info.npy')

    # Look into the info if the background noise is in the good interval
    mask_bkg = ((info[:,7] >= background_level - background_tolerance) & (info[:,7] <= background_level + background_tolerance) & (info[:,8] >= background_level - background_tolerance) & (info[:,8] <= background_level + background_tolerance) & (np.array([info[:,5] >=4, info[:,6] >=4]).any(axis=0)))
    if np.any(mask_bkg):
        info_to_add = info[mask_bkg]
        traces_to_add = traces[mask_bkg.astype(bool)]

        valid_info = np.append(valid_info, info_to_add, axis=0)
        valid_traces = np.append(valid_traces, traces_to_add, axis=0)

for i in range(valid_traces.shape[0]):
    valid_info[i,0] = i  # line number

print(f'Number of valid traces selected: {valid_traces.shape[0]}')

# Now that the traces and info are validates, we selected a third to save as is, acting as noise dataset
number_of_traces = valid_traces.shape[0]
dataset_size = number_of_traces // 3

print("Number of traces per dataset: ", dataset_size)


# We shuffle all of them
# indices = np.arange(number_of_traces)
# np.random.shuffle(indices)

# indices_noise = indices[:dataset_size]
# indices_ANhm = indices[dataset_size:2*dataset_size]
# indices_test = indices[2*dataset_size:]

# If we shuffle the events rather than the traces
indices_event = np.unique(valid_info[:, 1:3], axis=0)
np.random.shuffle(indices_event)
indices = np.array([])

for i in range(indices_event.shape[0]):
    event_indices = np.int32(valid_info[(valid_info[:,1] == indices_event[i,0]) & (valid_info[:,2] == indices_event[i,1])][:,0])
    indices = np.int32(np.append(indices, event_indices))

indices_noise = indices[:dataset_size]
indices_ANhm = indices[dataset_size:2*dataset_size]
indices_test = indices[2*dataset_size:]


# -------------------------------------------------
# We save the noise dataset
noise_traces = valid_traces[indices_noise]
noise_info = valid_info[indices_noise]

noise_traces_save = np.empty((0,512,2), dtype=np.int16)
noise_info_save = np.empty((0,11), dtype=np.float32)
t = np.linspace(0, 2048, 1024)
for i in range(noise_traces.shape[0]):
    for ch in range(3):
        noise_traces[i,ch,:] = noise_traces[i,ch,:] - np.mean(noise_traces[i,ch,:])
 
    # Since we know the position of the pulse (around the 150th point), we assume the pulse is shorter than 300 ns, and take 75 points both sides from the maximum
    # We slide the window randomly around the 150 points taken by the pulse, and in case we want the pulse at the end of the trace, we put the end of the trace (last 512 points) at the beginning
    n_max = np.argmax(np.sqrt(noise_traces[i,0,:]**2 + noise_traces[i,1,:]**2)) - 75
    # print(n_max)
    # n_min = np.argmax(np.sqrt(noise_traces[i,0,:]**2 + noise_traces[i,1,:]**2)) + 75 - 512
    n_min = 0
    if np.max([0, n_max]) == 0:
        n=0
    else:
        n = np.random.randint(n_min, np.max([0, n_max]))
    # print("n_min = ", n_min, " n_max = ", n_max)
    if n>=0:
        traces = noise_traces[i,:2,n:n+512]
    else:
        traces = np.append(noise_traces[i,:2,n:], noise_traces[i,:2,:n+512], axis=1)
    
    # # normalization :
    # traces[0] = traces[0] / noise_info[i,7]
    # traces[1] = traces[1] / noise_info[i,8]

    if n > 350 :
        continue
    noise_traces_save = np.append(noise_traces_save, [traces.T], axis=0)
    noise_info_save = np.append(noise_info_save, noise_info[i].reshape((1,11)), axis=0)

print(f'Saved noise dataset with {noise_traces_save.shape} traces.')
np.save(f'{save_noise_path}_{noise_traces_save.shape[0]}_traces.npy', noise_traces_save)
np.save(f'{save_noise_path}_{noise_traces_save.shape[0]}_info.npy', noise_info_save)




# -------------------------------------------------
# We build and save the handmade AN dataset

ANhm_traces = valid_traces[indices_ANhm]
ANhm_info = valid_info[indices_ANhm]

ANhm_traces_save = np.empty((0,512,2), dtype=np.int16)
ANhm_info_save = np.empty((0,11), dtype=np.float32)

# import NJ sims to add to the traces
repert_NJ = sorted(glob(f'{master_path}/gathered_traces/sims_NJ/adc*'))
list_NJ_traces = np.empty((0, 2, 1024), dtype=np.int16)
list_NJ_info = np.empty((0, 11), dtype=np.float32)
for file in repert_NJ:
    data = np.load(f'{file}/traces.npy')
    info = np.load(f'{file}/info.npy')
    traces = data[:, :2, :]
    list_NJ_traces = np.append(list_NJ_traces, traces, axis=0)
    list_NJ_info = np.append(list_NJ_info, info, axis=0)

    # trying not to get all traces saved up for problem of memory
    if list_NJ_traces.shape[0] >= (dataset_size*7):
        break

print(f'Loaded {list_NJ_traces.shape[0]} NJ sim traces to add to handmade AN dataset.')


last_NJ_sim = 0
NJ_sim_count = 0

for i in range(ANhm_traces.shape[0]):
    for ch in range(3):
        ANhm_traces[i,ch,:] = ANhm_traces[i,ch,:] - np.mean(ANhm_traces[i,ch,:])
    
    # it can happen that the pulse is at the end of the trace, so we look for the maximum, and exit if the maximum is after the 512th point
    if np.argmax(np.sqrt(ANhm_traces[i,0,:]**2 + ANhm_traces[i,1,:]**2)) > 512:
        continue


    # mask_pulse = (np.sqrt(ANhm_traces[i,0,:]**2 + ANhm_traces[i,1,:]**2)> 3)
    # pulse_size = t[mask_pulse][-1] - t[mask_pulse][0]
    # if pulse_size >= 512:
    #     continue



    # assume the pulse is shorter than 160 ns, and take 40 points both sides from the maximum
    pulse_size = 80

    # randomizing the position of the pulse in the 512-sample window
    n = np.random.randint(0 , 512 - pulse_size)
    # n=0
    # print("n = ", n)
    traces = ANhm_traces[i,:2,512:].copy()
    pos_max = np.argmax(np.sqrt(list_NJ_traces[NJ_sim_count,0,:]**2 + list_NJ_traces[NJ_sim_count,1,:]**2))
    # # bugged, but working version:
    # pos_max = np.argmax(np.sqrt(ANhm_traces[i,0,:]**2 + ANhm_traces[i,1,:]**2))

    SNR_X = np.max(list_NJ_traces[NJ_sim_count,0,:])/ANhm_info[i,6]
    SNR_Y = np.max(list_NJ_traces[NJ_sim_count,1,:])/ANhm_info[i,7]

    for k in range(2):
        # print(traces[k,n:n+pulse_size].shape)
        # print(list_NJ_traces[i,k,pos_max - pulse_size//2:pos_max + pulse_size//2].shape)
        traces[k,n:n+pulse_size] += list_NJ_traces[NJ_sim_count,k,pos_max - pulse_size//2:pos_max + pulse_size//2]
        
    # we make sure the pulse fits in the 1024-sample window
    while pos_max - pulse_size//2 <0 or pos_max - pulse_size//2 >1024 or SNR_X <4 or SNR_Y <4 or SNR_X > 70 or SNR_Y > 70:
        NJ_sim_count += 1
        pos_max = np.argmax(np.sqrt(list_NJ_traces[NJ_sim_count,0,:]**2 + list_NJ_traces[NJ_sim_count,1,:]**2))

        SNR_X = np.max(list_NJ_traces[NJ_sim_count,0,:])/ANhm_info[i,6]
        SNR_Y = np.max(list_NJ_traces[NJ_sim_count,1,:])/ANhm_info[i,7]

        traces = ANhm_traces[i,:2,512:].copy()
        for k in range(2):
            # print(traces[k,n:n+pulse_size].shape)
            # print(list_NJ_traces[i,k,pos_max - pulse_size//2:pos_max + pulse_size//2].shape)
            traces[k,n:n+pulse_size] += list_NJ_traces[NJ_sim_count,k,pos_max - pulse_size//2:pos_max + pulse_size//2]

    if not pass_T1(traces):
        # print("Did not pass T1: ", NJ_sim_count)
        NJ_sim_count += 1
        continue

    # build the info file
    info = ANhm_info[i].copy()
    info[0] = i  # line number
    info[1] = np.float64(str(np.int64(ANhm_info[i,1])) + '.' + str(np.int64(list_NJ_info[NJ_sim_count,1])))  # file names
    info[2] = np.int32(str(np.int64(ANhm_info[i,2])) + '0' + str(np.int64(list_NJ_info[NJ_sim_count,2])))  # root index for both parts
    info[3] = list_NJ_info[NJ_sim_count,3]  # number of antennas triggered
    info[4] = list_NJ_info[NJ_sim_count,4] # du id
    info[5] = SNR_X  # SNR X
    info[6] = SNR_Y  # SNR Y
    info[7] = ANhm_info[i,7]  # sigma stationnary X
    info[8] = ANhm_info[i,8]  # sigma stationnary Y
    info[9] = list_NJ_info[NJ_sim_count,9] # zenith theta
    info[10] = list_NJ_info[NJ_sim_count,10] # azimuth theta

    # # normalization :
    # traces[0] = traces[0] / ANhm_info[i,7]
    # traces[1] = traces[1] / ANhm_info[i,8]

    ANhm_traces_save = np.append(ANhm_traces_save, [np.swapaxes(traces, 0, 1)], axis=0)
    ANhm_info_save = np.append(ANhm_info_save, info.reshape((1,11)), axis=0)

    NJ_sim_count += 1
    last_NJ_sim += 1

print(f'Saved handmade AN dataset with {ANhm_traces_save.shape} traces.')
np.save(f'{save_ANhm_path}_{ANhm_traces_save.shape[0]}_traces.npy', ANhm_traces_save)
np.save(f'{save_ANhm_path}_{ANhm_info_save.shape[0]}_info.npy', ANhm_info_save)


# # -------------------------------------------------
# # We save the test dataset

# test_traces = valid_traces[indices_test]
# test_info = valid_info[indices_test]

# test_traces_save = np.empty((0,512,2), dtype=np.int16)
# test_info_save = np.empty((0,11), dtype=np.float32)
# t = np.linspace(0, 2048, 1024)
# for i in range(test_traces.shape[0]):
#     for ch in range(3):
#         test_traces[i,ch,:] = test_traces[i,ch,:] - np.mean(test_traces[i,ch,:])

#     # mask_pulse = (np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2)> 3)
#     # pulse_size = t[mask_pulse][-1] - t[mask_pulse][0]
#     # if pulse_size >= 512:
#     #     continue

#     n_max = np.argmax(np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2)) - 75
#     n_min = np.argmax(np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2)) + 75 - 512
#     n = np.random.randint(n_min, n_max)
#     # print("n_min = ", n_min, " n_max = ", n_max)
#     if n>=0:
#         traces = test_traces[i,:2,n:n+512]
#     else:
#         traces = np.append(test_traces[i,:2,n:], test_traces[i,:2,:n+512], axis=1)

#     # normalization :
#     traces[0] = traces[0] / test_info[i,7]
#     traces[1] = traces[1] / test_info[i,8]

#     if n > 350 :
#         continue
#     test_traces_save = np.append(test_traces_save, [traces.T], axis=0)
#     test_info_save = np.append(test_info_save, test_info[i].reshape((1,11)), axis=0)

# print(f'Saved test dataset with {test_traces_save.shape} traces.')

# np.save(f'{save_test_path}_{test_traces_save.shape[0]}_traces.npy', test_traces_save)
# np.save(f'{save_test_path}_{test_info_save.shape[0]}_info.npy', test_info_save)

# -------------------------------------------------
# Alternative for test dataset: half noise (like earlier), half AN handmade

test_traces = valid_traces[indices_test]
test_info = valid_info[indices_test]

test_traces_noise_save = np.empty((0,512,2), dtype=np.int16)
test_info_noise_save = np.empty((0,11), dtype=np.float32)
test_traces_ANhm_save = np.empty((0,512,2), dtype=np.int16)
test_info_ANhm_save = np.empty((0,11), dtype=np.float32)
t = np.linspace(0, 2048, 1024)
for i in range(test_traces.shape[0]//2):
    for ch in range(3):
        test_traces[i,ch,:] = test_traces[i,ch,:] - np.mean(test_traces[i,ch,:])

    n_max = np.argmax(np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2)) - 75
    # n_min = np.argmax(np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2)) + 75 - 512
    n_min = 0
    if np.max([0, n_max]) == 0:
        n=0
    else:
        n = np.random.randint(n_min, np.max([0, n_max]))

    if n>=0:
        traces = test_traces[i,:2,n:n+512]
    else:
        traces = np.append(test_traces[i,:2,n:], test_traces[i,:2,:n+512], axis=1)

    # # normalization :
    # traces[0] = traces[0] / test_info[i,7]
    # traces[1] = traces[1] / test_info[i,8]

    if n > 350 :
        continue
    test_traces_noise_save = np.append(test_traces_noise_save, [traces.T], axis=0)
    test_info_noise_save = np.append(test_info_noise_save, test_info[i].reshape((1,11)), axis=0)


NJ_sim_count += last_NJ_sim
for i in range(test_traces.shape[0]//2, test_traces.shape[0]):
    for ch in range(3):
        test_traces[i,ch,:] = test_traces[i,ch,:] - np.mean(test_traces[i,ch,:])
    
    # assume the pulse is shorter than 160 ns, and take 40 points both sides from the maximum
    pulse_size = 80

    n = np.random.randint(0 , 512 - pulse_size)
    traces = test_traces[i,:2,512:].copy()
    pos_max = np.argmax(np.sqrt(list_NJ_traces[NJ_sim_count,0,:]**2 + list_NJ_traces[NJ_sim_count,1,:]**2))
    # # bugged, but working version:
    # pos_max = np.argmax(np.sqrt(test_traces[i,0,:]**2 + test_traces[i,1,:]**2))

    SNR_X = np.max(list_NJ_traces[NJ_sim_count,0,:])/test_info[i,6]
    SNR_Y = np.max(list_NJ_traces[NJ_sim_count,1,:])/test_info[i,7]

    for k in range(2):
        # print(traces[k,n:n+pulse_size].shape)
        # print(list_NJ_traces[i,k,pos_max - pulse_size//2:pos_max + pulse_size//2].shape)
        traces[k,n:n+pulse_size] += list_NJ_traces[NJ_sim_count,k,pos_max - pulse_size//2:pos_max + pulse_size//2]
        
    # we make sure the pulse fits in the 1024-sample window
    while pos_max - pulse_size//2 <0 or pos_max - pulse_size//2 >1024 or SNR_X <4 or SNR_Y <4 or SNR_X > 70 or SNR_Y > 70:
        NJ_sim_count += 1
        pos_max = np.argmax(np.sqrt(list_NJ_traces[NJ_sim_count,0,:]**2 + list_NJ_traces[NJ_sim_count,1,:]**2))

        SNR_X = np.max(list_NJ_traces[NJ_sim_count,0,:])/test_info[i,6]
        SNR_Y = np.max(list_NJ_traces[NJ_sim_count,1,:])/test_info[i,7]

        traces = test_traces[i,:2,512:].copy()
        for k in range(2):
            # print(traces[k,n:n+pulse_size].shape)
            # print(list_NJ_traces[i,k,pos_max - pulse_size//2:pos_max + pulse_size//2].shape)
            traces[k,n:n+pulse_size] += list_NJ_traces[NJ_sim_count,k,pos_max - pulse_size//2:pos_max + pulse_size//2]

    if not pass_T1(traces):
        # print("Did not pass T1: ", NJ_sim_count)
        NJ_sim_count += 1
        continue
    
    # build the info file
    info = test_info[i].copy()
    info[0] = i  # line number
    info[1] = np.float64(str(np.int64(test_info[i,1])) + '.' + str(np.int64(list_NJ_info[NJ_sim_count,1])))  # file names
    info[2] = np.int32(str(np.int64(test_info[i,2])) + '0' + str(np.int64(list_NJ_info[NJ_sim_count,2])))  # root index for both parts
    info[3] = list_NJ_info[NJ_sim_count,3]  # number of antennas triggered
    info[4] = list_NJ_info[NJ_sim_count,4] # du id
    info[5] = SNR_X  # SNR X
    info[6] = SNR_Y  # SNR Y
    info[7] = test_info[i,7]  # sigma stationnary X
    info[8] = test_info[i,8]  # sigma stationnary Y
    info[9] = list_NJ_info[NJ_sim_count,9] # zenith theta
    info[10] = list_NJ_info[NJ_sim_count,10] # azimuth theta

    # # normalization :
    # traces[0] = traces[0] / test_info[i,7]
    # traces[1] = traces[1] / test_info[i,8]

    test_traces_ANhm_save = np.append(test_traces_ANhm_save, [np.swapaxes(traces, 0, 1)], axis=0)
    test_info_ANhm_save = np.append(test_info_ANhm_save, info.reshape((1,11)), axis=0)
    NJ_sim_count += 1

print(f'Saved test dataset (half noise, half AN handmade) with {test_traces_noise_save.shape[0] + test_traces_ANhm_save.shape[0]} traces.')
np.save(f'{save_test_path}_noise_{test_traces_noise_save.shape[0]}_traces.npy', test_traces_noise_save)
np.save(f'{save_test_path}_noise_{test_info_noise_save.shape[0]}_info.npy', test_info_noise_save)
np.save(f'{save_test_path}_ANhm_{test_traces_ANhm_save.shape[0]}_traces.npy', test_traces_ANhm_save)
np.save(f'{save_test_path}_ANhm_{test_info_ANhm_save.shape[0]}_info.npy', test_info_ANhm_save)
    
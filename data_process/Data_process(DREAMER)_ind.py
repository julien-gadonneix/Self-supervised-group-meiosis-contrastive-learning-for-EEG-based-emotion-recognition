# import os
import numpy as np
# import time
# import scipy.io as scio
import torch
import torch.nn.functional as F
from pathlib import Path
import scipy.io

import mne

sub_to_remove = 1 # between 0 and 22

random_seed = 0
np.random.seed(random_seed)
window_size = 128

def windows(data, size):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += size

# Tag reading
cur_dir = Path(__file__).resolve().parent.parent
dataset_dir = str(cur_dir) + '/data/DREAMER/'
mat = scipy.io.loadmat(dataset_dir + 'DREAMER.mat')
data, eeg_sr, ecg_sr, _, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
n_subjects = int(n_subjects[0, 0])
n_videos = int(n_videos[0, 0])

DATA = []
LABEL_aro = []
LABEL_val = []
LABEL_dom = []
DATA_ind = []
LABEL_aro_ind = []
LABEL_val_ind = []
LABEL_dom_ind = []

for i in range(n_videos): # Movie loop
    video_data = []
    video_aro = []
    video_val = []
    video_dom = []
    for j in range(n_subjects): # Video loop
        if j == sub_to_remove:
            continue
        _, _, eeg, ecg, val, aro, dom = data[0, j][0][0]
        baseline_eeg, stimuli_eeg = eeg[0, 0]
        stimuli_eeg_j = stimuli_eeg[i, 0]
        baseline_eeg_j = baseline_eeg[i, 0]
        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, .5, None, method='iir',
                                                iir_params=dict(order=3, rp=0.1, rs=60, ftype='butter'), verbose=False)
        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, .5, None, method='iir',
                                                iir_params=dict(order=3, rp=0.1, rs=60, ftype='butter'), verbose=False)
        baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], 128)
        avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
        std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
        stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], 128)[1:]
        stimulis_j -= avg_baseline_eeg_j
        stimulis_j /= std_baseline_eeg_j
        stimulis_j = stimulis_j.transpose(1, 0, 2)
        stimulis_j = stimulis_j.reshape(stimulis_j.shape[0], -1)
        val_label = (val[i, 0] > 3) * 1 # val[i, 0] - 1
        aro_label = (aro[i, 0] > 3) * 1 # aro[i, 0] - 1
        dom_label = (dom[i, 0] > 3) * 1 # dom[i, 0] - 1
        print('val:', val_label, 'aro:', aro_label, 'dom:', dom_label, 'sub:', j, 'videos:',i)
        video_data.append(stimulis_j)
        video_aro.append(aro_label)
        video_val.append(val_label)
        video_dom.append(dom_label)
    data_in = np.array(video_data)
    label_aro = np.array(video_aro)
    label_val = np.array(video_val)
    label_dom = np.array(video_dom)
    for (start, end) in windows(data_in[0,0], window_size):
        if ((len(data_in[0,0,start:end]) == window_size)):
            segments = data_in[:,:,start:end]
            segments = F.normalize(torch.tensor(segments), dim=2)
            segments = segments.numpy()
            DATA.append(segments)
            LABEL_aro.append(label_aro)
            LABEL_val.append(label_val)
            LABEL_dom.append(label_dom)
            print(len(DATA), len(LABEL_aro))

DATA = np.array(DATA)
DATA = np.expand_dims(DATA, -3)
# LABEL_aro = np.array(LABEL_aro)
# LABEL_val = np.array(LABEL_val)
# LABEL_dom = np.array(LABEL_dom)
LABEL = np.column_stack((LABEL_aro, LABEL_val, LABEL_dom))

for i in range(n_videos): # Movie loop
    video_data = []
    video_aro = []
    video_val = []
    video_dom = []
    _, _, eeg, ecg, val, aro, dom = data[0, sub_to_remove][0][0]
    baseline_eeg, stimuli_eeg = eeg[0, 0]
    stimuli_eeg_j = stimuli_eeg[i, 0]
    baseline_eeg_j = baseline_eeg[i, 0]
    stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, .5, None, method='iir',
                                            iir_params=dict(order=3, rp=0.1, rs=60, ftype='butter'), verbose=False)
    baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, .5, None, method='iir',
                                            iir_params=dict(order=3, rp=0.1, rs=60, ftype='butter'), verbose=False)
    baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], 128)
    avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
    std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
    stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], 128)[1:]
    stimulis_j -= avg_baseline_eeg_j
    stimulis_j /= std_baseline_eeg_j
    stimulis_j = stimulis_j.transpose(1, 0, 2)
    stimulis_j = stimulis_j.reshape(stimulis_j.shape[0], -1)
    val_label = (val[i, 0] > 3) * 1 # val[i, 0] - 1
    aro_label = (aro[i, 0] > 3) * 1 # aro[i, 0] - 1
    dom_label = (dom[i, 0] > 3) * 1 # dom[i, 0] - 1
    print('val:', val_label, 'aro:', aro_label, 'dom:', dom_label, 'sub:', sub_to_remove, 'videos:',i)
    video_data.append(stimulis_j)
    video_aro.append(aro_label)
    video_val.append(val_label)
    video_dom.append(dom_label)
    data_in = np.array(video_data)
    label_aro = np.array(video_aro)
    label_val = np.array(video_val)
    label_dom = np.array(video_dom)
    for (start, end) in windows(data_in[0,0], window_size):
        if ((len(data_in[0,0,start:end]) == window_size)):
            segments = data_in[:,:,start:end]
            segments = F.normalize(torch.tensor(segments), dim=2)
            segments = segments.numpy()
            DATA_ind.append(segments)
            LABEL_aro_ind.append(label_aro)
            LABEL_val_ind.append(label_val)
            LABEL_dom_ind.append(label_dom)
            print(len(DATA_ind), len(LABEL_aro_ind))

DATA_ind = np.array(DATA_ind)
DATA_ind = np.expand_dims(DATA_ind, -3)
# LABEL_aro = np.array(LABEL_aro)
# LABEL_val = np.array(LABEL_val)
# LABEL_dom = np.array(LABEL_dom)
LABEL_ind = np.column_stack((LABEL_aro_ind, LABEL_val_ind, LABEL_dom_ind))

#Get Training , Test and validation set
from sklearn.model_selection import train_test_split
x_train_DREAMER, x_test_DREAMER, y_train_DREAMER, y_test_DREAMER = train_test_split(DATA, LABEL, random_state=42, test_size=0.2)
np.save(f'x_train_DREAMER_ind({sub_to_remove}).npy',x_train_DREAMER)
np.save(f'x_test_DREAMER_ind({sub_to_remove}).npy',x_test_DREAMER)
np.save(f'x_val_DREAMER_ind({sub_to_remove}).npy',DATA_ind)

np.save(f'y_train_DREAMER_ind({sub_to_remove}).npy',y_train_DREAMER)
np.save(f'y_test_DREAMER_ind({sub_to_remove}).npy',y_test_DREAMER)
np.save(f'y_val_DREAMER_ind({sub_to_remove}).npy',LABEL_ind)
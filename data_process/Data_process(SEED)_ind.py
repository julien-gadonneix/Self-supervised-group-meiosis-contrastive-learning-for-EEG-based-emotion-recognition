import os
import numpy as np
# import time
import scipy.io as scio
import torch
import torch.nn.functional as F
from pathlib import Path

sub_to_remove = 1 # between 1 and 15

random_seed = 0
np.random.seed(random_seed)
window_size = 200

def windows(data, size):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += size

# Tag reading
cur_dir = Path(__file__).resolve().parent.parent
dataset_dir = str(cur_dir) + '/data/SEED'
record_list_ori = sorted([task for task in os.listdir(dataset_dir) if task.endswith('.mat') and task!='label.mat'])
record_list = [record for record in record_list_ori if record.split('_')[0] != str(sub_to_remove)]
record_list_ind = [record for record in record_list_ori if record.split('_')[0] == str(sub_to_remove)]
labels_root = os.path.join(dataset_dir, 'label.mat')
labels = scio.loadmat(labels_root)['label'][0] + 1

DATA = []
LABEL = []
DATA_ind = []
LABEL_ind = []

# Get name of video
keys_array = []
for record in record_list:
    print(record)
    file = os.path.join(dataset_dir, record)
    keys = list(scio.loadmat(file).keys())[3:]
    keys_array.append(keys)
keys_array = np.array(keys_array)
# keys_array.shape
# for i in range(keys_array.shape[1]):
#     sample_file = os.path.join(dataset_dir, record_list[0])
#     sample_mat = scio.loadmat(sample_file)
#     sample_keys = list(sample_mat.keys())[3:]
# Take a file as a baseline first
for i in range(keys_array.shape[1]): # Movie loop
    t = -1
    video_data = []
    video_label = []
    label = labels[i]
    for j in range(len(record_list)): # File loop (session*subject)
        t += 1
        sub = int(t/3)
        times = t%3
        file_name = record_list[j]
        sub_video_name = keys_array[j,i]
        # label = labels[i]
        print('label:', label, 'sub:', sub, 'times:',times, 'file_name:',file_name, 'sub_video_name:',sub_video_name)
        file_root = os.path.join(dataset_dir, file_name)
        file_data = scio.loadmat(file_root)
        data_tem = file_data[sub_video_name] # [channel]
        # label_tem = label #, sub, times, i]
        video_data.append(data_tem)
        video_label.append(label)
    data_in = np.array(video_data)
    label_in = np.array(video_label)
    for (start, end) in windows(data_in[0,0], window_size):
        if ((len(data_in[0,0,start:end]) == window_size)):
            segments = data_in[:,:,start:end]
            segments = F.normalize(torch.tensor(segments), dim=2)
            segments = segments.numpy()
            # label, sub, times, video, start
            # labels = label_in # np.column_stack((label_in, np.array([start for k in range(label_in.shape[0])])))
            DATA.append(segments)
            LABEL.append(label_in)
            print(len(DATA), len(LABEL))

DATA = np.array(DATA)
DATA = np.expand_dims(DATA,-3)
LABEL = np.array(LABEL)

# Get name of video
keys_array = []
for record in record_list_ind:
    print(record)
    file = os.path.join(dataset_dir, record)
    keys = list(scio.loadmat(file).keys())[3:]
    keys_array.append(keys)
keys_array = np.array(keys_array)
# keys_array.shape
# for i in range(keys_array.shape[1]):
#     sample_file = os.path.join(dataset_dir, record_list_ind[0])
#     sample_mat = scio.loadmat(sample_file)
#     sample_keys = list(sample_mat.keys())[3:]
# Take a file as a baseline first
for i in range(keys_array.shape[1]): # Movie loop
    t = -1
    video_data = []
    video_label = []
    label = labels[i]
    for j in range(len(record_list_ind)): # File loop (session*subject)
        t += 1
        sub = int(t/3)
        times = t%3
        file_name = record_list_ind[j]
        sub_video_name = keys_array[j,i]
        # label = labels[i]
        print('label:', label, 'sub:', sub, 'times:',times, 'file_name:',file_name, 'sub_video_name:',sub_video_name)
        file_root = os.path.join(dataset_dir, file_name)
        file_data = scio.loadmat(file_root)
        data_tem = file_data[sub_video_name] # [channel]
        # label_tem = label #, sub, times, i]
        video_data.append(data_tem)
        video_label.append(label)
    data_in = np.array(video_data)
    label_in = np.array(video_label)
    for (start, end) in windows(data_in[0,0], window_size):
        if ((len(data_in[0,0,start:end]) == window_size)):
            segments = data_in[:,:,start:end]
            segments = F.normalize(torch.tensor(segments), dim=2)
            segments = segments.numpy()
            # label, sub, times, video, start
            # labels = label_in # np.column_stack((label_in, np.array([start for k in range(label_in.shape[0])])))
            DATA_ind.append(segments)
            LABEL_ind.append(label_in)
            print(len(DATA_ind), len(LABEL_ind))

DATA_ind = np.array(DATA_ind)
DATA_ind = np.expand_dims(DATA_ind,-3)
LABEL_ind = np.array(LABEL_ind)

#Get Training , Test and validation set
from sklearn.model_selection import train_test_split
x_train_SEED, x_test_SEED, y_train_SEED, y_test_SEED = train_test_split(DATA, LABEL, random_state=42, test_size=0.3)
# np.save('DATA_SEED.npy',DATA)
# np.save('LABEL_SEED.npy',LABEL)
np.save(f'x_train_SEED_ind({sub_to_remove}).npy',x_train_SEED)
np.save(f'x_test_SEED_ind({sub_to_remove}).npy',x_test_SEED)
np.save(f'x_val_SEED_ind({sub_to_remove}).npy',DATA_ind)
np.save(f'y_train_SEED_ind({sub_to_remove}).npy',y_train_SEED)
np.save(f'y_test_SEED_ind({sub_to_remove}).npy',y_test_SEED)
np.save(f'y_val_SEED_ind({sub_to_remove}).npy',LABEL_ind)
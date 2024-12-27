import numpy as np
# import os
# from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
# from torch.autograd import Variable
# import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models.ResNet_model import ResNet
# import pandas as pd

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device: ", device)
is_ok = device.type!='mps'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Test function
def _eval(model, test_loader, loss_fn, is_ok):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for pair in test_loader:
            x, y = pair[0], pair[1]
            # x = x.cuda().float().contiguous()
            # y = y.cuda().long().contiguous()
            x = x.to(device=device, memory_format=torch.channels_last, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            if is_ok:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    out = model(x, mode='classifier')
                    loss = loss_fn(out, y)
            else:
                out = model(x, mode='classifier')
                loss = loss_fn(out, y)
            acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float()/y.shape[0]).item()
            loss = loss.item()
            total_loss += loss * y.shape[0]
            total_acc += acc * y.shape[0]
        test_loss = total_loss / len(test_loader.dataset)
        test_acc = total_acc / len(test_loader.dataset)
    return test_loss, test_acc

#Training function
def _train(model, train_loader, optimizer, scaler, epoch, loss_fn, is_ok):
    model.train()
    train_losses_tem = []
    train_acces_tem = []
    for pair in train_loader:
        x, y = pair[0], pair[1]
        # x = x.cuda().float().contiguous()
        # y = y.cuda().long().contiguous()
        x = x.to(device=device, memory_format=torch.channels_last, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        if is_ok:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(x, mode='classifier')
                loss = loss_fn(out, y)
        else:
            out = model(x, mode='classifier')
            loss = loss_fn(out, y)
        acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float() / y.shape[0]).item()
        scaler.scale(loss).backward()
        loss = loss.item()
        scaler.step(optimizer)
        scaler.update()
        train_losses_tem.append(loss)
        train_acces_tem.append(acc)
    print(f'Epoch: {epoch}, Train_loss: {np.mean(train_losses_tem):.4f}, Train_acc: {np.mean(train_acces_tem):.4f}')
    return train_losses_tem, train_acces_tem

#Emotional recognition training function
def _train_epochs(model, train_loader, test_loader, epochs, lr, device, is_ok):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=is_ok)
    loss_fn = nn.CrossEntropyLoss().to(device)
    train_losses = []
    train_acces = []
    test_loss, test_acc = _eval(model, test_loader, loss_fn, is_ok)
    test_losses = [test_loss]
    test_acces = [test_acc]
    print(f'Epoch {0}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')
    # Begin training
    for epoch in range(1, epochs+1):
        train_losses_tem, train_acces_tem = _train(model, train_loader, optimizer, scaler, epoch, loss_fn, is_ok)
        #print(f'Epoch: {epoch}, batch: {num}, Train_loss: {loss:.4f}, Train_acc: {acc:.4f}')
        train_acces.extend(train_acces_tem)
        train_losses.extend(train_losses_tem)
        test_loss, test_acc = _eval(model, test_loader, loss_fn, is_ok)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        print(f'Epoch {epoch}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')
    return train_losses, train_acces, test_losses, test_acces

sub_to_remove = 1
#Import data and set hyper-parameters
x_train_path = 'x_train_SEED.npy'
x_test_path = 'x_val_SEED.npy'
y_train_path = 'y_train_SEED.npy'
y_test_path = 'y_val_SEED.npy'
# x_train_path = 'x_train_DREAMER.npy'
# x_test_path = 'x_val_DREAMER.npy'
# y_train_path = 'y_train_DREAMER.npy'
# y_test_path = 'y_val_DREAMER.npy'
# x_train_path = 'x_train_SEED_ind(1).npy'
# x_test_path = 'x_val_SEED_ind(1).npy'
# y_train_path = 'y_train_SEED_ind(1).npy'
# y_test_path = 'y_val_SEED_ind(1).npy'
# x_train_path = f'x_train_DREAMER_ind({sub_to_remove}).npy'
# x_test_path = f'x_val_DREAMER_ind({sub_to_remove}).npy'
# y_train_path = f'y_train_DREAMER_ind({sub_to_remove}).npy'
# y_test_path = f'y_val_DREAMER_ind({sub_to_remove}).npy'

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)

# n_sub_train = x_train.shape[1]
# n_sub_test = x_test.shape[1]

x_train = x_train.reshape(-1, 1, x_train.shape[-2], x_train.shape[-1])
x_test = x_test.reshape(-1, 1, x_test.shape[-2], x_test.shape[-1])
# y_train_aro, y_train_val, y_train_dom = np.split(y_train, 3, axis=1)
# y_train = y_train_val.reshape(-1,)
# y_test_aro, y_test_val, y_test_dom = np.split(y_test, 3, axis=1)
# y_test = y_test_val.reshape(-1,)
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# y_train = np.tile(y_train, reps=(n_sub_train,))
# y_test = np.tile(y_test, reps=(n_sub_test,))

batch_size = 256
train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

epochs = 300 # 50 # 60 # 70, 1000 (stop at 600)
lr = 0.001 # 0.001, 0.00001

#Import pre-trained model
# ssl_root = 'Pretrained_ResNet_DEAP;totalepochs=4000;P=008;Q=02;tau=1.00E-01;lr=1.00E-04//Pretrained_ResNet_DEAP;totalepochs=4000;P=008;Q=02;tau=1.00E-01;lr=1.00E-04;epoch=0940.pth'
ssl_root = 'Pretrained_ResNet_SEED;totalepochs=3288;P=016;Q=02;tau=1.00E-01;lr=1.00E-03/Pretrained_ResNet_SEED;totalepochs=3288;P=016;Q=02;tau=1.00E-01;lr=1.00E-03;epoch=3288.pth'
# ssl_root = 'Pretrained_ResNet_DREAMER;totalepochs=2800;P=008;Q=02;tau=1.00E-01;lr=1.00E-04/Pretrained_ResNet_DREAMER;totalepochs=2800;P=008;Q=02;tau=1.00E-01;lr=1.00E-04;epoch=2800.pth'
# ssl_root = f'Pretrained_ResNet_DREAMER_ind({sub_to_remove});totalepochs=2800;P=008;Q=02;tau=1.00E-01;lr=1.00E-04/Pretrained_ResNet_DREAMER_ind({sub_to_remove});totalepochs=2800;P=008;Q=02;tau=1.00E-01;lr=1.00E-04;epoch=2800.pth'
model = ResNet(num_classes=3) # 2 # 5
model = torch.load(ssl_root)
model.to(device=device, memory_format=torch.channels_last)

#Training
torch.backends.cudnn.benchmark = True
_, _, _, acces = _train_epochs(model, train_loader, test_loader, epochs, lr, device, is_ok)

saved_models_dir = ssl_root.split('/')[0]
ssl_logs = os.path.join(saved_models_dir,'ssl_logs')
# name = f'FineTuning;epochs={epochs};lr={lr};batch_size={batch_size}.jpg'
name = f'FineTuning;epochs={epochs};lr={lr};batch_size={batch_size}.jpg'

#Plot
import matplotlib.pyplot as plt
plt.plot(acces)
plt.legend(['fine-tune_ssl'], loc='upper left')
plt.ylabel('test_acc')
plt.xlabel('epoch')
plt.title(f'Fine-tuning with {acces[-1]} accuracy')
plt.savefig(os.path.join(ssl_logs, name))
plt.show()

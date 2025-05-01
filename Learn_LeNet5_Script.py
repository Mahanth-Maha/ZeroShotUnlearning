# # LeNet5
# * Author : Mahanth Yalla
# * Sr No  : 24004

# ### imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset

import os
import tqdm
import math
import json
import pickle
import pynvml
import argparse
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime as dt
import matplotlib.gridspec as gridspec


def get_best_gpu(verbose = False):
    pynvml.nvmlInit()
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        if verbose :
            print("[!] No CUDA devices found, running on CPU.")
        return 'cpu'

    gpu_mem = []
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free
        total_mem = mem_info.total
        if verbose:
            print(f"[+] GPU {i}: Total {total_mem / 1e9:.2f} GB, Free {free_mem / 1e9:.2f} GB")
        gpu_mem.append((i, free_mem))

    gpu_mem.sort(key=lambda x: x[1], reverse=True)

    best_gpu = gpu_mem[0][0]
    second_best_gpu = gpu_mem[1][0] if len(gpu_mem) > 1 else best_gpu

    selected_gpu = second_best_gpu if best_gpu == 0 else best_gpu

    pynvml.nvmlShutdown()
    return selected_gpu

def set_gpu(manual_set = None, verbose = False):
    device_active = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_active == 'cuda':
        if verbose:
            print('[+] CUDA : available')
        try:
            if manual_set is not None:
                GPU_NUMBER = manual_set
            else:
                GPU_NUMBER = get_best_gpu(verbose)
            if verbose:
                print(f'[>] Current Device {torch.cuda.current_device()} -> changing it to {GPU_NUMBER}')
            torch.cuda.set_device(GPU_NUMBER)
            if verbose:
                print(f'[+] Current Device {torch.cuda.current_device()}')
        except Exception as e:
            if verbose:
                print(f'[-] Error Occured while changing GPU ! \n\n\tERROR :{e}\n\n')
                print(f'[!] CUDA ERR : Couldn\'t change  -> using CPU {device_active = }')
        finally: 
            pass
        print(f'[>] device = cuda:{GPU_NUMBER}')
    else :
        if verbose:
            print(f'[!] CUDA ERR : Not available -> using CPU {device_active = }')
        print(f'[>] device = cpu')

SCRIPT_start_time = dt.now()

parser = argparse.ArgumentParser(description='LeNet5 CIFAR-10 Training Script')
parser.add_argument('-g','--gpu_number', type=int, default=None, help='GPU number to use')

parser.add_argument('-b','--batch_size', type=int, default=64, help='batch size : defaults to 64')
parser.add_argument('-d','--dropout', type=float, default=0.15, help='dropout value')

parser.add_argument('-e','--total_epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('-v','--eval_every', type=int, default=10, help='eval K times in a epochs')
parser.add_argument('-s','--save_every', type=int, default=10, help='save after every K epochs')
parser.add_argument('-f','--forget_class', type=int, default=10, help='forget class number from 0 - 9 (def = 10)')
parser.add_argument('-n','--version_name', type=str, default="v0_test", help='version name for saving models')
parser.add_argument('--net_name', type=str, default="LeNet5", help='Net name for saving models')

args = parser.parse_args()

set_gpu(args.gpu_number)

forget_class = args.forget_class
version_name = args.version_name
dir_model = f'./models/{args.net_name}_{version_name}'


mnist_data_folder = './data/MNIST/'
labels = np.arange(10)
mnist_train_images_file =  mnist_data_folder + f'train-images.idx3-ubyte'
mnist_train_labels_file =  mnist_data_folder + f'train-labels.idx1-ubyte'
mnist_test_images_file =  mnist_data_folder + f't10k-images.idx3-ubyte'
mnist_test_labels_file =  mnist_data_folder + f't10k-labels.idx1-ubyte'

# Reading from idx file function code snippet is taken from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40#file-mnist-py
def read_idx(filename):
    import struct 
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

mnist_train_images = read_idx(mnist_train_images_file)
mnist_train_labels = read_idx(mnist_train_labels_file)
mnist_test_images = read_idx(mnist_test_images_file)
mnist_test_labels = read_idx(mnist_test_labels_file)
mnist_train_images.shape ,mnist_train_labels.shape,mnist_test_images.shape ,mnist_test_labels.shape

def train_test_split(X,y,test_size = 0.2 , shuffle = True):
    if shuffle:
        mask = np.random.permutation(len(y))
        X = X[mask]
        y = y[mask]
    split = len(y) - int(len(y) * test_size)
    return X[:split],y[:split],X[split:],y[split:]

Xtr, ytr, Xval, yval = train_test_split(mnist_train_images, mnist_train_labels, test_size=0.2)
Xte , yte = mnist_test_images, mnist_test_labels

print(f'MNIST DATASET :')
print(f'{Xtr.shape = } , {ytr.shape = }')
print(f'{Xval.shape = } , {yval.shape = }')
print(f'{Xte.shape = } , {yte.shape = }')

Xtr = torch.tensor(Xtr , dtype = torch.float32)
ytr = torch.tensor(ytr , dtype = torch.float32)
Xval = torch.tensor(Xval , dtype = torch.float32)
yval = torch.tensor(yval , dtype = torch.float32)
Xte = torch.tensor(Xte , dtype = torch.float32)
yte = torch.tensor(yte , dtype = torch.float32)

ytr_enc = torch.zeros(ytr.shape[0] , len(labels)) 
yval_enc = torch.zeros(yval.shape[0] , len(labels))
yte_enc = torch.zeros(yte.shape[0] , len(labels))

ytr_enc[torch.arange(ytr.shape[0]) , ytr.long()] = 1
yval_enc[torch.arange(yval.shape[0]) , yval.long()] = 1
yte_enc[torch.arange(yte.shape[0]) , yte.long()] = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    Xtr = Xtr.cuda()
    Xval = Xval.cuda()
    Xte = Xte.cuda()
    ytr_enc = ytr_enc.cuda()
    yval_enc = yval_enc.cuda()
    yte_enc = yte_enc.cuda()
    ytr = ytr.cuda()
    yval = yval.cuda()
    yte = yte.cuda()    


class DatasetMNIST(Dataset):
    def __init__(self , X , y , labels = labels, device = device):
        self.X,self.y = X, y
        self.num_classes = len(labels)
        self.labels = np.sort(np.arange(self.num_classes))
        self.device = device
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self , idx):
        pixs = np.random.randint(1,4)
        X_i , y_i = self.random_agumentation(self.X[idx] , self.y[idx] , pixels = pixs)
        y_i = F.one_hot(y_i.long() , num_classes = self.num_classes).float()
        
        if device == 'cuda':
            X_i = X_i.cuda()
            y_i = y_i.cuda()
        
        return X_i , y_i
    
    def random_agumentation(self , X_i , y_i , pixels = 1):
        type = np.random.randint(1,2 * 8 + 1) # half of the time : original
        if type == 1:
            X_Right_shifted = torch.roll(X_i , shifts = pixels , dims = 1)
            X_shifted = X_Right_shifted
        elif type == 2:
            X_Left_shifted = torch.roll(X_i , shifts = -pixels , dims = 1)
            X_shifted = X_Left_shifted
        elif type == 3:
            X_Up_shifted = torch.roll(X_i , shifts = pixels , dims = 0)
            X_shifted = X_Up_shifted
        elif type == 4:
            X_Down_shifted = torch.roll(X_i , shifts = -pixels , dims = 0)
            X_shifted = X_Down_shifted
        elif type == 5:
            X_Right_shifted = torch.roll(X_i , shifts = pixels , dims = 1)
            X_shifted = torch.roll(X_Right_shifted , shifts = pixels , dims = 0)
        elif type == 6:
            X_Left_shifted = torch.roll(X_i , shifts = -pixels , dims = 1)
            X_shifted = torch.roll(X_Left_shifted , shifts = -pixels , dims = 0)
        elif type == 7:
            X_Up_shifted = torch.roll(X_i , shifts = pixels , dims = 0)
            X_shifted = torch.roll(X_Up_shifted , shifts = -pixels , dims = 1)
        elif type == 8:
            X_Down_shifted = torch.roll(X_i , shifts = -pixels , dims = 0)
            X_shifted = torch.roll(X_Down_shifted , shifts = pixels , dims = 1)
        else:
            X_shifted = X_i
        
        return X_shifted , y_i
    
NUM_EPOCHS = args.total_epochs
BATCH_SIZE = args.batch_size
DROPOUT    = args.dropout

EVAL_EVERY = (Xtr.shape[0] // BATCH_SIZE) // args.eval_every ### evaluate 10 (X) times per epoch
SAVE_EVERY = (Xtr.shape[0] // BATCH_SIZE) *  args.save_every ### save after 10 (K) epochs

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-7
BETAS = (0.9, 0.999)

LABEL_SMOOTHING = 0.1
T_MAX_SCHEDULER = (Xtr.shape[0] // BATCH_SIZE) // 100


class LeNet5(nn.Module):
    
    def __init__(self, inp_size = 1, filters = (6 , 16), f_sizes = (5,5) , num_classes = 10):
        super(LeNet5,self).__init__() 
        
        self.conv1 = nn.Conv2d(inp_size, filters[0], kernel_size = f_sizes[0], stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size = f_sizes[1], stride = 1, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(filters[1]*f_sizes[1]*f_sizes[1], 120 , bias = True )
        self.fc2 = nn.Linear(120, 84 , bias = True )
        self.fc3 = nn.Linear(84, num_classes, bias = True )
        
    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    

    
    def predict_proba(self, x):
        x = self.forward(x)
        return F.softmax(x, dim = -1)

    def predict(self, x):
        x = self.predict_proba(x)
        return torch.argmax(x, dim = -1)
    
    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return (y_pred == y).float().mean()
    
    
# Xb , yb = next(iter(dataloader))
# print(f'{Xb.shape = } , {yb.shape = }')
# lenet5 = LeNet5()
# lenet5.forward(Xb.reshape(-1,1,28,28)).shape
# lenet5.forward(Xb.reshape(-1,1,28,28))[0].shape

# def smax(x):
#     return torch.exp(x) / torch.exp(x).sum(axis = -1)

# smax(lenet5.forward(Xb.reshape(-1,1,28,28))[0])

# torch.softmax(lenet5.forward(Xb.reshape(-1,1,28,28)) , dim = -1)[0] , yb[0]

# ## LeNet5 Trainer

class LeNetTrainer():
    ''' Trainer class to train the model with training and validation dataset'''
    def __init__(self,directory, model, optimizer, loss, train_dataset_loader, valid_dataset_loader, scheduler = None, device = 'cpu'):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        
        self.directory = directory
        self.device = device
        self.last_checkpoint = 0
        
        self.train_dataset_loader = train_dataset_loader
        self.valid_dataset_loader = valid_dataset_loader

        self.logs = {
            'train':[],
            'valid':[]
            }
        
        if device == 'cuda':
            self.model = self.model.cuda()
            print('[+] Model is moved to GPU')

        os.makedirs(self.directory, exist_ok=True)
        
    def train(self, num_epochs=10, batch_size=8, save_steps=100, eval_steps=100):
        current_checkpoint = 0
        self.model.to(self.device)

        with tqdm.tqdm(total = math.ceil(len(train_dataset) / batch_size) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                for batch, (x_batch, y_batch) in enumerate(self.train_dataset_loader):
                    pbar.set_description(f"Epoch {epoch+1} / {num_epochs}")

                    if current_checkpoint < self.last_checkpoint:
                        current_checkpoint += 1
                        pbar.update()
                        continue

                    loss = self.training(x_batch, y_batch)
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                    self.logs['train'].append(loss)
                    pbar.set_postfix({ 'batch': batch+1, 'loss': loss })
                    current_checkpoint += 1
                    pbar.update()

                    if (current_checkpoint) % eval_steps == 0:
                        val_loss = self.validation()
                        self.logs['valid'].append(val_loss)
                        
                        print('[T]', f"epoch #{epoch+1:{len(str(num_epochs))}},",f"batch #{batch+1:{len(str(len(self.train_dataset_loader)))}}:",
                            f"loss: {loss:.8f} | val_loss: {val_loss:.8f}")

                    if (current_checkpoint) % save_steps == 0:
                        self.save(current_checkpoint, { 'loss': loss, 'checkpoint': current_checkpoint })

            self.save(current_checkpoint)

    def training(self, Xb , yb):
        ''' training step for a single batch'''
        self.model.train()
        
        y_pred = self.model(Xb)
        loss = self.loss(y_pred, yb)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() / len(yb)
   
    def validation(self):
        ''' validation step for a single epoch'''
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for Xb, yb in self.valid_dataset_loader:
                y_pred = self.model(Xb)
                loss = self.loss(y_pred, yb)
                total_loss += loss.item()
        return total_loss /  (len(self.valid_dataset_loader) * self.valid_dataset_loader.batch_size)
    
    def resume(self):
        """ Resumes training session from the most recent checkpoint. """

        if checkpoints := os.listdir(self.directory):
            self.last_checkpoint = max(map(lambda x: int(x[11:]), filter(lambda x: 'checkpoint-' in x, checkpoints)))
            checkpoint_dir = os.path.join(self.directory, f"checkpoint-{self.last_checkpoint}")
            self.model.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                map_location=self.device
            ))
            self.model.to(self.device)
            self.optimizer.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "optimizer.pt"),
                map_location=self.device
            ))
            with open(os.path.join(checkpoint_dir, "loss.json"), 'r', encoding='utf-8') as ifile:
                self.logs = json.load(ifile)

    def save(self, checkpoint=None, metadata=None):
        """ Saves an associated model or a training checkpoint."""

        if checkpoint is not None:
            checkpoint_dir = os.path.join(self.directory, f"checkpoint-{checkpoint}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
            with open(os.path.join(checkpoint_dir, "loss.json"), "w+", encoding='utf-8') as ofile:
                json.dump(self.logs, ofile, ensure_ascii=False, indent=2)
            if metadata:
                with open(os.path.join(checkpoint_dir, "metadata.json"), "w+", encoding='utf-8') as ofile:
                    json.dump(metadata, ofile, ensure_ascii=False, indent=2)
        else:
            torch.save(self.model, os.path.join(self.directory, "model.pt"))
            with open(os.path.join(self.directory, "loss.json"), "w+", encoding='utf-8') as ofile:
                json.dump(self.logs, ofile, ensure_ascii=False, indent=2)
            if metadata:
                with open(os.path.join(self.directory, "metadata.json"), "w+", encoding='utf-8') as ofile:
                    json.dump(metadata, ofile, ensure_ascii=False, indent=2)

    def plot_losses(self, save_path = None):
        ''' plot the training and validation losses'''
        plt.figure(figsize=(10,5))
        plt.plot(self.logs['train'],label='Train Loss')
        plt.plot(self.logs['valid'],label='Valid Loss')
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
            return
        plt.show()
    
    def accuracy(self, X, y):
        ''' calculate the accuracy of the model on the dataset given'''
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X.reshape(-1,1,28,28))
            return (y_pred == y).float().mean()
    
    def predict(self, X):
        ''' predict the class labels for the input dataset X'''
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(X.reshape(-1,1,28,28))

    
net_params = {
    'inp_size': 1,
    'filters': (6, 16),
    'f_sizes': (5, 5),
    'num_classes': len(labels)
}

  
net_training_params = dict(
    num_epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE, 
    save_steps=SAVE_EVERY, 
    eval_steps=EVAL_EVERY
)
  

model = LeNet5(**net_params)
print(f'MODEL :\n{model}')
print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()
train_dataset = DatasetMNIST(Xtr , ytr , device = device)
valid_dataset = DatasetMNIST(Xval , yval , device = device)

train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

trainer = LeNetTrainer(
    dir_model,
    model, 
    optimizer, 
    loss, 
    train_dataset_loader, 
    valid_dataset_loader , 
    scheduler=None,
    device=device,
)


before_training_acc = trainer.accuracy(Xval, yval)
print(f'[>] Validation Accuracy before training: {before_training_acc:.2%}')

# ### Training

time_start = dt.now()

trainer.resume()

trainer.train(
    **net_training_params
)

# trainer.plot_losses()

trainer.save(metadata={
    'model'   : net_params,
    'training': net_training_params
})

print(f'[>] Training time: {dt.now() - time_start}')

# ### Evaluation

# trainer.plot_losses()

after_training_acc = trainer.accuracy(Xval, yval)
print(f'[>] Validation Accuracy after training: {after_training_acc:.2%}')

# ### Test Accuracy Report
Test_Accuracy = trainer.accuracy(Xte, yte)
print(f'[*] Test Accuracy: {Test_Accuracy:.2%}')

SCRIPT_end_time = dt.now()
print(f'[>] Script Execution Time : {SCRIPT_end_time - SCRIPT_start_time}')
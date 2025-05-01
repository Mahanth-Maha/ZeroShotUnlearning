# * **Title** : Learn KarpathyNet with Forgetting a class
# * **Author** : Yalla Mahanth
# * **SR No** : 24004
# 

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

parser = argparse.ArgumentParser(description='KarpathyNet CIFAR-10 Training Script with Forgetting a class')
parser.add_argument('-g','--gpu_number', type=int, default=None, help='GPU number to use')

parser.add_argument('-b','--batch_size', type=int, default=64, help='batch size : defaults to 64')
parser.add_argument('-d','--dropout', type=float, default=0.15, help='dropout value')

parser.add_argument('-e','--total_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('-v','--eval_every', type=int, default=10, help='eval K times in a epochs')
parser.add_argument('-s','--save_every', type=int, default=10, help='save after every K epochs')
parser.add_argument('-f','--forget_class', type=int, default=10, help='forget class number from 0 - 9 (def = 10)')
parser.add_argument('-n','--version_name', type=str, default="v0_test", help='version name for saving models')
parser.add_argument('--net_name', type=str, default="KarpathyNet", help='Net name for saving models')

args = parser.parse_args()

set_gpu(args.gpu_number)

forget_class = args.forget_class
version_name = args.version_name
dir_model = f'./models/{args.net_name}_{version_name}'

root_ = './'
cifar_data_folder = f'{root_}data/cifar10/'

cifar_train_batch_1_file = cifar_data_folder + 'data_batch_1'
cifar_train_batch_2_file = cifar_data_folder + 'data_batch_2'
cifar_train_batch_3_file = cifar_data_folder + 'data_batch_3'
cifar_train_batch_4_file = cifar_data_folder + 'data_batch_4'
cifar_train_batch_5_file = cifar_data_folder + 'data_batch_5'
cifar_test_file = cifar_data_folder + 'test_batch' 
cifar_meta_data = cifar_data_folder + 'batches.meta' 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_batch_1 = unpickle(cifar_train_batch_1_file) 
cifar_batch_2 = unpickle(cifar_train_batch_2_file)
cifar_batch_3 = unpickle(cifar_train_batch_3_file)
cifar_batch_4 = unpickle(cifar_train_batch_4_file)
cifar_batch_5 = unpickle(cifar_train_batch_5_file)
cifar_batch_test = unpickle(cifar_test_file)
batches_meta = unpickle(cifar_meta_data)

cifar_batch_1_data = cifar_batch_1[b'data']
cifar_batch_1_labels = np.array(cifar_batch_1[b'labels'])
cifar_batch_2_data = cifar_batch_2[b'data']
cifar_batch_2_labels = np.array(cifar_batch_2[b'labels'])
cifar_batch_3_data = cifar_batch_3[b'data']
cifar_batch_3_labels = np.array(cifar_batch_3[b'labels'])
cifar_batch_4_data = cifar_batch_4[b'data']
cifar_batch_4_labels = np.array(cifar_batch_4[b'labels'])
cifar_batch_5_data = cifar_batch_5[b'data']
cifar_batch_5_labels = np.array(cifar_batch_5[b'labels'])

cifar_batch_test_data = cifar_batch_test[b'data']
cifar_batch_test_labels = np.array(cifar_batch_test[b'labels'])

cifar_train_data = np.vstack([cifar_batch_1_data, cifar_batch_2_data, cifar_batch_3_data,cifar_batch_4_data, cifar_batch_5_data]).reshape(-1,32*32*3)
cifar_train_labels = np.hstack([cifar_batch_1_labels, cifar_batch_2_labels, cifar_batch_3_labels,cifar_batch_4_labels, cifar_batch_5_labels])

cifar_test_data = cifar_batch_test_data.reshape(-1,32*32*3)
cifar_test_labels = cifar_batch_test_labels

labels = np.unique(cifar_train_labels)

cifar_label_dict = {}
for i, label in enumerate(labels):
    cifar_label_dict[label] = batches_meta[b'label_names'][i].decode('utf-8')

def train_test_split(X,y,test_size = 0.2 , shuffle = True):
    if shuffle:
        mask = np.random.permutation(len(y))
        X = X[mask]
        y = y[mask]
    split = len(y) - int(len(y) * test_size)
    return X[:split],y[:split],X[split:],y[split:]

def remove_class(X,y,forget_class = 10):
    mask = y != forget_class
    return X[mask],y[mask]

cifar_train_data, cifar_train_labels = remove_class(cifar_train_data, cifar_train_labels, forget_class)
Xtr,  ytr, Xval, yval = train_test_split(cifar_train_data, cifar_train_labels, test_size=0.2)
Xte , yte = cifar_test_data, cifar_test_labels
label_dict = cifar_label_dict



# if forget_class in labels:
#     print('[>] Restricting the dataset by forgetting the class :', forget_class)
#     # labels = labels[labels != forget_class]
#     Xtr = Xtr[ytr != forget_class]
#     ytr = ytr[ytr != forget_class]
    
#     Xval = Xval[yval != forget_class]
#     yval = yval[yval != forget_class]

    

print(f'[>] CIFAR-10 Dataset')
print(f'{Xtr.shape = }\t{ytr.shape = }')
print(f'{Xval.shape = }\t{yval.shape = }')
print(f'{Xte.shape = }\t{yte.shape = }')
print(f'{label_dict = }')

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

Xtr = Xtr.view(-1,3,32,32)
Xval = Xval.view(-1,3,32,32)
Xte = Xte.view(-1,3,32,32)

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



class DatasetCifar(Dataset):
    def __init__(self , X , y, labes, device = device, img_shape = (3,32,32)):
        self.X,self.y = X, y
        self.labels = torch.sort(torch.tensor(labels))
        self.n_classes = len(labels)
        self.each_img_shape = img_shape
        self.device = device
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self , idx):
        pixs = np.random.randint(1,4)
        X_i = self.random_agumentation(self.X[idx] ,  pixels = pixs)
        y_i = F.one_hot(self.y[idx].long() , num_classes = self.n_classes).float()
        
        if self.device == 'cuda':
            X_i = X_i.cuda()
            y_i = y_i.cuda()
        
        return X_i , y_i
        
    
    def rotate_tensor(self, img, angle):
        angle = torch.tensor(angle * np.pi / 180)
        theta = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0]])
        if self.device == 'cuda':
            angle = angle.cuda()
            theta = theta.cuda()

        theta = theta.unsqueeze(0)
        grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False )
        rotated_img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return rotated_img.squeeze(0)
    
    def random_agumentation(self , X_i , pixels = 1):
        X_i = X_i.reshape(self.each_img_shape)
        aug_type = np.random.randint(0,30) + 1 # [>] half of the time : original
        if aug_type == 1:       # [>] right shift
            X_Right_shifted = torch.roll(X_i , shifts = pixels , dims = 2)
            X_shifted = X_Right_shifted
        elif aug_type == 2:     # [>] left shift
            X_Left_shifted = torch.roll(X_i , shifts = -pixels , dims = 2)
            X_shifted = X_Left_shifted
        elif aug_type == 3:     # [>] up shift
            X_Up_shifted = torch.roll(X_i , shifts = pixels , dims = 1)
            X_shifted = X_Up_shifted
        elif aug_type == 4:     # [>] down shift
            X_Down_shifted = torch.roll(X_i , shifts = -pixels , dims = 1)
            X_shifted = X_Down_shifted
        elif aug_type == 5:     # [>] NE shift
            X_Right_shifted = torch.roll(X_i , shifts = pixels , dims = 2)
            X_shifted = torch.roll(X_Right_shifted , shifts = pixels , dims = 1)
        elif aug_type == 6:     # [>] NW shift
            X_Left_shifted = torch.roll(X_i , shifts = -pixels , dims = 2)
            X_shifted = torch.roll(X_Left_shifted , shifts = -pixels , dims = 1)
        elif aug_type == 7:     # [>] SE shift
            X_Up_shifted = torch.roll(X_i , shifts = pixels , dims = 1)
            X_shifted = torch.roll(X_Up_shifted , shifts = -pixels , dims = 2)
        elif aug_type == 8:     # [>] SW shift
            X_Down_shifted = torch.roll(X_i , shifts = -pixels , dims = 1)
            X_shifted = torch.roll(X_Down_shifted , shifts = pixels , dims = 2)
        elif aug_type == 9:     # [>] flip ( left - right)
            X_flip = torch.flip(X_i , dims = [-1]) 
            X_shifted = X_flip
        elif  10 <= aug_type < 16: # [>] model sees roated more images
            angle = np.random.uniform(-45, 45)
            X_shifted = self.rotate_tensor(X_i, angle)
        else :
            X_shifted = X_i
        
        return X_shifted 
    
train_dataset = DatasetCifar(Xtr , ytr, labels)
len(train_dataset)

dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
Xb, yb = next(iter(dataloader))
Xb.shape, yb.shape


class Evaluator:
    """ Class to handle all the logic concerning the evaluation of trained models.  """
    def __init__(self, device):
        self.device = device

    @staticmethod
    def accuracy(y_true, y_pred):
        """ Computes the accuracy of the predictions, against a reference set of predictions. """
        return sum(yi_true == yi_pred for yi_true, yi_pred in zip(y_true, y_pred)) / len(y_pred)

    
    def evaluate(self, model_path, data, reference_outputs):
        """ Performs the evaluation of a specified model over given data. """
        model = torch.load(os.path.join(model_path, 'model.pt'), map_location=self.device, weights_only=False)
        model.to(self.device)
        model.eval()
        
        generated_outputs = model.predict(data)
        accuracy_score = self.accuracy(reference_outputs, generated_outputs)

        print("[E] EVALUATION:", ">", "accuracy:", f"{accuracy_score:.2%}")
        print()
    
    def evaluate_model(self, model, data, reference_outputs):
        """ Performs the evaluation of a specified model over given data. """
        model.to(self.device)
        model.eval()
        
        generated_outputs = model.predict(data)
        accuracy_score = self.accuracy(reference_outputs, generated_outputs)

        print("[E] EVALUATION:", ">", "accuracy:", f"{accuracy_score:.2%}")
        print()


# ### Visualizing the data

def plot_cifar(X,y,n = 10):
    """ Plot 10 random images from each class """
    plt.figure(figsize = (20,10))
    for i in range(n):
        mask = y == i 
        X_ = X[mask]
        y_ = y[mask]
        idx = np.random.randint(0,X_.shape[0])
        plt.subplot(1,n,i+1)
        plt.imshow(X_[idx].reshape(3,32,32).transpose(1,2,0))
        plt.title(cifar_label_dict[y_[idx]])
        plt.axis('off')
    plt.show()
# plot_cifar(cifar_train_data,cifar_train_labels)
# plot_cifar(cifar_train_data,cifar_train_labels)

def plot_img(Xb , yb , save_path = None):
    ''' plot a random image from the batch with its label (single image plot)'''
    idx = np.random.randint(0,Xb.shape[0])
    plt.figure(figsize = (2,2))
    plt.imshow(Xb[idx].cpu().numpy().astype(np.uint8).reshape(3,32,32).transpose(1,2,0))
    plt.title(cifar_label_dict[yb[idx].argmax().item()])
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return
    plt.show()

def plot_probs(sf_probs ,label_dict = cifar_label_dict , k = 5 , save_path = None):
    ''' plot top k probabilities in bars horizontally'''
    plt.figure(figsize=(5, 2))
    sorted_indices = torch.argsort(sf_probs, descending=True)
    sorted_probs = sf_probs[sorted_indices]
    sorted_labels = [label_dict[i.item()] for i in sorted_indices]

    plt.barh(sorted_labels[:k], sorted_probs.cpu().detach().numpy()[:k])
    plt.xlabel('Probability')
    plt.ylabel('Class')
    plt.title('Top 5 Softmax Probabilities')
    plt.gca().invert_yaxis()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return
    plt.show()

def plot_img_with_probs(Xb, yb, model, label_dict=cifar_label_dict, k=5, save_path = None):
    ''' plot a random image from the batch with its label and top k probabilities side by side'''
    idx = np.random.randint(0, Xb.shape[0]) 
    
    img = Xb[idx].cpu().numpy().astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    true_label_idx = yb[idx].argmax().item()
    true_label = cifar_label_dict[true_label_idx]
    
    outputs = model(Xb[idx].unsqueeze(0))
    sf_probs = F.softmax(outputs, dim=1)
    
    sorted_indices = torch.argsort(sf_probs[0], descending=True)
    sorted_probs = sf_probs[0][sorted_indices]
    sorted_labels = [label_dict[i.item()] for i in sorted_indices]
    colors = ['green' if sorted_indices[i].item() == true_label_idx else 'red' for i in range(k)]

    fig = plt.figure(figsize=(6, 2))  
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1]) 

    ax0 = plt.subplot(gs[0])
    ax0.imshow(img)
    ax0.set_title(f"Label: {true_label}")
    ax0.axis('off')
    ax1 = plt.subplot(gs[1:])
    ax1.barh(sorted_labels[:k], sorted_probs.cpu().detach().numpy()[:k], color=colors)
    ax1.set_xlabel('Probability')
    ax1.set_title('Top 5 Predictions')
    ax1.invert_yaxis()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return
    plt.show()

def plot_img_with_probs_bottom(Xb, yb, model, label_dict=cifar_label_dict, k=5 ,save_path =None):
    ''' plot a random image from the batch with its label and top k probabilities top and bottom'''
    idx = np.random.randint(0, Xb.shape[0])
    
    img = Xb[idx].cpu().numpy().astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    true_label_idx = yb[idx].argmax().item()  
    true_label = cifar_label_dict[true_label_idx] 
    
    outputs = model(Xb[idx].unsqueeze(0))
    sf_probs = F.softmax(outputs, dim=1)
    
    sorted_indices = torch.argsort(sf_probs[0], descending=True)
    sorted_probs = sf_probs[0][sorted_indices]
    sorted_labels = [label_dict[i.item()] for i in sorted_indices]

    colors = ['green' if sorted_indices[i].item() == true_label_idx else 'red' for i in range(k)]

    fig, axes = plt.subplots(2, 1, figsize=(3,5), gridspec_kw={'height_ratios': [2, 1]})

    axes[0].imshow(img)
    axes[0].set_title(f"Label: {true_label}")
    axes[0].axis('off')

    axes[1].barh(sorted_labels[:k], sorted_probs.cpu().detach().numpy()[:k], color=colors)
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Top 5 Predictions')
    axes[1].invert_yaxis()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        return
    plt.show()

def compare_with_images(model, Xte, yte, lables = 10 , device = 'cpu', save_path = None):
    ''' plot 10 images from each class and compare the actual and predicted labels (mark with green-tick or red-cross)'''
    fig = plt.figure(figsize=(10,2))
    for i in range(10):
        ax = fig.add_subplot(1, lables, i+1)
        imgs = Xte[yte == i]
        img_i = np.random.randint(imgs.shape[0])
        model_pred = model.predict(imgs[img_i] , single_input = True)
        lbls = yte[yte == i]
        
        if device == 'cuda':
            plotting_img = imgs[img_i].cpu().numpy().astype(np.uint8).reshape(3,32,32).transpose(1,2,0)
        else:
            plotting_img = imgs[img_i].cpu().numpy().astype(np.uint8).reshape(3,32,32).transpose(1,2,0)
        ax.imshow(plotting_img)
        
        if model_pred[0] == lbls[img_i]:
            ax.set_title(f'Actual :{int(lbls[img_i].cpu().numpy().astype(np.uint8))} \nPredict :{int(model_pred[0])} $\\checkmark$ \nimage:', color='green', fontsize= 10)
        else:
            ax.set_title(f'Actual :{int(lbls[img_i].cpu().numpy().astype(np.uint8))} \nPredict :{int(model_pred[0])} $\\times$ \nimage:', color='red', fontsize= 9)
        
        ax.axis('off')
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    plt.show()

# # CifarClassification

class KarpathyNet(nn.Module):
    
    def __init__(self, inp_size, filters, filter_sizes, strides, paddings , num_classes, kernal_size = 2, stride = 2):
        super(KarpathyNet,self).__init__() 
        
        self.conv11 = nn.Conv2d(inp_size, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.conv12 = nn.Conv2d(filters, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.pool1 = nn.MaxPool2d(kernel_size = kernal_size, stride = stride)
        self.conv21 = nn.Conv2d(filters, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.conv22 = nn.Conv2d(filters, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.pool2 = nn.MaxPool2d(kernel_size = kernal_size, stride = stride)
        self.conv31 = nn.Conv2d(filters, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.conv32 = nn.Conv2d(filters, filters, kernel_size = filter_sizes, stride = strides, padding = paddings)
        self.pool3 = nn.MaxPool2d(kernel_size = kernal_size, stride = stride)
        self.fc_input = 4*4*filters
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc_input, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool2(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
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

class Trainer():
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
                            f"loss: {loss:.8f} | val_loss: {val_loss:.8f} | lr : {self.scheduler.get_last_lr()[0]:.8f}")

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
            y_pred = self.model.predict(X)
            return (y_pred == y).float().mean()
    
    def predict(self, X):
        ''' predict the class labels for the input dataset X'''
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(X)


# ### Training


net_params = {
    'inp_size': 3,
    'filters': 16,
    'filter_sizes': 5,
    'strides': 1,
    'paddings': 2,
    'num_classes': len(labels)
}


net_training_params = dict(
    num_epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE, 
    save_steps=SAVE_EVERY, 
    eval_steps=EVAL_EVERY
)

sr_no = 24004
torch.manual_seed(sr_no)

model = KarpathyNet(**net_params)
print(f'MODEL :\n{model}')
print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE , weight_decay = WEIGHT_DECAY , betas = BETAS)
print(f'optimizer : AdamW\n {optimizer}')

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX_SCHEDULER)
print(f'Scheduler : CosineAnnealingLR')

loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
print(f'Loss : Cross Entropy Loss')

train_dataset = DatasetCifar(Xtr , ytr ,labels, device = device)
valid_dataset = DatasetCifar(Xval , yval , labels, device = device)

train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

trainer = Trainer(
    dir_model,
    model, 
    optimizer, 
    loss, 
    train_dataset_loader, 
    valid_dataset_loader , 
    scheduler=scheduler,
    device=device,
)
evaluator = Evaluator(device)

trainer.resume()

trainer.train(
    **net_training_params
)

trainer.plot_losses()

evaluator.evaluate_model(model,Xte, yte)

trainer.save(metadata={
    'model'   : net_params,
    'training': net_training_params
})

evaluator.evaluate(dir_model,Xte, yte)

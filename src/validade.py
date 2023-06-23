

import numpy as np
import matplotlib.pyplot as plt
import vgg_unet
import torch
from data_utils import SegmentationDataset
from data_utils import mixup_batch
from torch.utils.data import DataLoader, ConcatDataset
import cv2
from torchsummary import summary
from torch.utils.data import DataLoader
import glob
import random
import os.path as osp

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os, shutil


import json

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gt_folder_val = './Validation'
plot_val = True
work_resolution = (512, 384)
resolution = (451, 288)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class_to_id = {
    'dontcare': 0,
    'Background': 1,
    'femur': 2,
    'inf': 3,
    'skin': 4,
    'sup': 5
}
id_to_class = {v: k for k, v in class_to_id.items()}

nClasses = 6

viridis = cm.get_cmap('viridis', nClasses)


model = vgg_unet.UNetVgg(nClasses, False).to(device)
model_name = 'segm.pth'


try:
    model.load_state_dict(torch.load(model_name))
except:
    print("Unable to load weigths")
    exit(-1)
summary(model, input_size=(3, resolution[0], resolution[1]))

model.eval()


validation_dataset = DataLoader(
    SegmentationDataset(
        gt_folder_val,
        gt_folder_val,
        False,
        class_to_id,
        resolution=resolution
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
total_accuracy = []
per_classe_accuracy = {
    'dontcare': [1],
    'Background': [],
    'femur': [],
    'inf': [],
    'skin': [],
    'sup': []
}

all_preds = []
all_gts = []
with torch.no_grad():
    for i_batch, sample_batched in enumerate(validation_dataset):
        image = sample_batched['image'].to(device)
        gt = sample_batched['gt'].to(device)
        output = model(image)
        all_preds.append(output.argmax(1).cpu().numpy())
        all_gts.append(gt.cpu().numpy())
        accuracy = (output.argmax(1) == gt).float().mean()
        total_accuracy.append(accuracy.cpu().numpy().item())
        for i in range(1, 6):
            per_classe_accuracy[id_to_class[i]].append(
                ((output.argmax(1) == i) == (gt==i)).float().mean().cpu().numpy().item()
            )
        
del per_classe_accuracy['dontcare']

print(len(total_accuracy))
per_classe_accuracy = {k: np.mean(v) * 100 for k, v in per_classe_accuracy.items()}
print("Per class accuracy:\n")
for k, v in per_classe_accuracy.items():
    print(f"\t{k}: {v:.3f}")

all_preds = np.stack(all_preds).reshape(-1)
all_gts = np.stack(all_gts).reshape(-1)
dont_care = np.where(all_gts == 0)[0]
all_gts[dont_care] = 1
print(f'\nTotal Accuracy: {np.mean(all_gts == all_preds):.3f}\n')

from sklearn.metrics import classification_report
print(classification_report(all_gts, all_preds, labels=[1,2,3,4,5], target_names=['Background', 'femur', 'inf', 'skin', 'sup']))

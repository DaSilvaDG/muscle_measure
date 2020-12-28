

import numpy as np
import matplotlib.pyplot as plt
import vgg_unet
import torch
from data_utils import SegmentationDataset
from data_utils import mixup_batch
from torch.utils.data import DataLoader, ConcatDataset
import cv2
from torchsummary import summary

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
gt_folder_train = './Training/'
gt_folder_val = './Validation'
patience = 2000
plot_val = True
plot_train = False
work_resolution = (512, 384)
resolution = (451, 288)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


list_files = glob.glob('./Validation/**/*.tif*', recursive=True)
list_files += glob.glob('./Validation/**/*.jpg*', recursive=True)


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
with torch.no_grad():
    for file_ in list_files:

        print(file_)
        img_np = cv2.imread(file_,
                            cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        if not isinstance(img_np, np.ndarray):
            print('Unable to open file %s' %(file_))
        
        img_pt = img_np.astype(np.float32) / np.max(img_np)
        for i in range(3):
            img_pt[..., i] -= mean[i]
            img_pt[..., i] /= std[i]

        img_pt = np.expand_dims(img_pt.transpose(2, 0, 1), axis=0)
    
        label_out = model(torch.from_numpy(img_pt).float().to(device))
        label_out = torch.nn.functional.softmax(label_out, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)

        labels = np.argmax(label_out, axis=0).astype(np.uint8)
        for i in range(nClasses):
            contours, hierarchy = cv2.findContours((labels == i).astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[-2:]
            contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
            if contour:
                contour = contour[0]
                if i != 0 and i != 1:
                    epsilon = 0.01*cv2.arcLength(contour,True)
                    approx = cv2.approxPolyDP(contour,epsilon,True) 
                    cv2.drawContours(img_np, [approx], 0, viridis.colors[i,:3] * 255, 1)

        


        # img_np = img_np / 255.
        # for i in range(nClasses):
        #     img_np[labels == i] = img_np[labels == i] + viridis.colors[i,:3]*0.7
        # image_np = image_np * 255.
        cv2.imshow("Image", img_np)
        cv2.waitKey(0)











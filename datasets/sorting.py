###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Sorting Datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
import torch
import pandas as pd
import os

import ai8x
#from models.sortingnet import SortingClassifier128


'''
Sorting Dataset Class
Parameters:
  img_dir_path - Full path to directory with the images for this dataset.
                 This assumes that the subdirectories contain each class, 
                 only images are in these subdirectories, and that the
                 subdirectory basenames are the desired name of the object class.
                 i.e. dog/dog1.png, cat/cat1.png, etc.

  transform -    Specifies the image format (size, RGB, etc.) and augmentations to use
  normalize -    Specifies whether to make the image zero mean, unit variance
'''
class SortingDataset(Dataset):
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        print(self.classes)
        
        # get all training samples/labels by getting paths of the images in each subfolder folder
        self.imgs = []
        self.labels = []
        i = 0
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            if idx > 0: # we don't want the files in the top folder
                for file in path_obj[2]: # path_obj[2] is list of files in the subdirectory
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get the label from the directory name
                    i+=1
                    
    def read_img(self,img_path):
        w = 128
        h = 128
        c = 3
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        pixel_h = img_file.read(1)
        pixel_l = img_file.read(1)

        idx = 0
        while pixel_h:
            # extract the RGB values
            r = (pixel_h[0] & 0b11111000)>>3
            g = ((pixel_h[0] & 0b00000111)<<3) | ((pixel_l[0] & 0b11100000)>>5)
            b = pixel_l[0] & 0b00011111
            
            # get the x,y coordinate of the pixel
            x = idx%w
            y = idx//w
            
            # scale to RGB888 and save
            img[y,x,0] = (r<<3)
            img[y,x,1] = (g<<2)
            img[y,x,2] = (b<<3)
            idx += 1
            
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)
            
        return img#torch.from_numpy(img)

    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load the image
        try:
            img = self.read_img(self.imgs[idx])
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int)
            return img, label
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Displays a random batch of 64 samples
    def visualize_batch(self,model):
        import matplotlib
        matplotlib.use('TkAgg')
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        preds = model(imgs)
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.classes)
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = str(labels[idx].item()) + ":" + obj_classes[labels[idx]]  + "P: ",obj_classes[preds[idx].argmax()]#", i=" +str(idxs[idx].item())
                ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))
                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()
        
        
'''Function to get the datasets'''
def sorting_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    img_dir_path = "/home/geffen/Desktop/sorting_imgs/sorting_imgs4"

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.85,1.15)),#,saturation=(0.5,1),contrast=(0.7,1.1)),
            transforms.RandomGrayscale(0.10),
            transforms.RandomAffine(degrees=5,translate=(0.05,0.05)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        test_dataset = None
        
    # create the datasets
    train_dataset = SortingDataset(os.path.join(img_dir_path,"train"),train_transform,normalize=False)
    test_dataset = SortingDataset(os.path.join(img_dir_path,"test"),test_transform,normalize=False)
    
    return train_dataset, test_dataset


 #----------------------
 
 
class SortingDatasetBB(Dataset):
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        print(self.classes)
        
        # get all training samples/labels by getting paths of the images in each subfolder folder
        self.imgs = []
        self.labels = []
        i = 0
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            if idx > 0: # we don't want the files in the top folder
                for file in path_obj[2]: # path_obj[2] is list of files in the subdirectory
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get the label from the directory name
                    i+=1
                    
    def read_img(self,img_path):
        w = 128
        h = 128
        c = 3
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        pixel_h = img_file.read(1)
        pixel_l = img_file.read(1)

        idx = 0
        while pixel_h:
            # extract the RGB values
            r = (pixel_h[0] & 0b11111000)>>3
            g = ((pixel_h[0] & 0b00000111)<<3) | ((pixel_l[0] & 0b11100000)>>5)
            b = pixel_l[0] & 0b00011111
            
            # get the x,y coordinate of the pixel
            x = idx%w
            y = idx//w
            
            # scale to RGB888 and save
            img[y,x,0] = (r<<3)
            img[y,x,1] = (g<<2)
            img[y,x,2] = (b<<3)
            idx += 1
            
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)
            
        return img#torch.from_numpy(img)

    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load the image
        try:
            img = self.read_img(self.imgs[idx])
            file_name = os.path.basename(self.imgs[idx])+".png"
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # get the bb
            if label == 5:
                return img, torch.tensor([0,0,0,0]).float()
            
            df = pd.read_csv(os.path.dirname(self.img_dir_path) + "/" + list(self.classes)[label]+".csv")
            row = df.loc[df['filename'] == file_name]
            bb = [row['x'].item(),row['y'].item(),row['w'].item(),row['h'].item()]
            bb = torch.tensor(bb).float()
            
            # return the sample (img (tensor)), object class (int)
            return img, bb
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Displays a random batch of 64 samples
    def visualize_batch(self,model):
        import matplotlib
        matplotlib.use('TkAgg')
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        # get the first batch
        (imgs, bbs) = next(iter(data_loader))
        predictions = model(imgs)
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.classes)
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        print(predictions)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = "P: "#,obj_classes[preds[idx].argmax()]#", i=" +str(idxs[idx].item())
                
                preds = predictions[idx].detach()
                
                ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))
                rect_gt = patches.Rectangle((bbs[idx][0],bbs[idx][1]),bbs[idx][2],bbs[idx][3], edgecolor='r', facecolor="none")
                rect_pd = patches.Rectangle((preds[0].item(),preds[1].item()),preds[2].item(),preds[3].item(), edgecolor='g', facecolor="none")
                ax_array[i,j].add_patch(rect_gt)
                ax_array[i,j].add_patch(rect_pd)
                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()
        
        
'''Function to get the datasets'''
def sorting_get_datasetsbb(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    img_dir_path = "/home/geffen/Desktop/sorting_imgs_all/"

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.85,1.15)),#,saturation=(0.5,1),contrast=(0.7,1.1)),
            transforms.RandomGrayscale(0.10),
            transforms.RandomAffine(degrees=5,translate=(0.05,0.05)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        test_dataset = None
        
    # create the datasets
    train_dataset = SortingDatasetBB(os.path.join(img_dir_path,"train"),train_transform,normalize=False)
    test_dataset = SortingDatasetBB(os.path.join(img_dir_path,"test"),test_transform,normalize=False)
    
    return train_dataset, test_dataset


datasets = [
    {
        'name': 'sorting',
        'input': (3, 128, 128),
        'output': ('cup','hex','Trap','can','bottle','none'),
        'loader': sorting_get_datasets,
    },
    {
        'name': 'sortingbb',
        'input': (3, 128, 128),
        'output': ('cup','hex','Trap','can','bottle','none'),
        'loader': sorting_get_datasetsbb,
    }
]
    
    
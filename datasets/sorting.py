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
Datasets for sorting images
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
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import ai8x


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
        
        # get all training samples/labels by getting absolute paths of the images in each subfolder
        self.imgs = [] # absolute img paths (all images)
        self.labels = [] # integer labels (all labels in corresponding order)

        i = 0 # index into dataset lists

        # iterate through the dataset directory tree
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            # each execution of this inner loop is for each subdirectory
            if idx > 0: # don't include files in the top folder (subfolders are in the next itertion, idx > 0)
                for file in path_obj[2]: # path_obj[2] is list of files in the object class subdirectories
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get label from directory name
                    i+=1

    # since the image are in RGB565 we need a special function to read them into RGB888             
    def read_img(self,img_path):
        # img_test_path = os.path.join(os.path.basename(os.path.dirname(img_path)),os.path.basename(img_path))
        # test_img = False
        # if(img_test_path == "Plastic/img0190"):
        #     test_img = True

        # width, height, channel
        w = 128
        h = 128
        c = 3

        # images are 8 bits per channel
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        # each RGB565 pixel is two bytes
        #            High byte    Low byte
        # RGB565 --> RRRRRGGG     GGGBBBBB
        pixel_h = img_file.read(1)
        pixel_l = img_file.read(1)
        #print(img_path)
        idx = 0
        while pixel_h:
            # extract the RGB values
            r = (pixel_h[0] & 0b11111000)>>3
            g = ((pixel_h[0] & 0b00000111)<<3) | ((pixel_l[0] & 0b11100000)>>5)
            b = pixel_l[0] & 0b00011111
            
            # get the x,y coordinate of the pixel
            x = idx%w
            y = idx//w
            # if y > 127 or x > 127:
            #     # print(img_path)
            #     # print(idx)
            #     # print(pixel_h)
            #     break
            # scale to RGB888 and save
            img[y,x,0] = (r<<3)
            img[y,x,1] = (g<<2)
            img[y,x,2] = (b<<3)
            idx += 1
            
            # try to read the next pixel
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)

        # if test_img == True:
        #     # save the img after converting to rgb888  
        #     np.save("/home/geffen_cooper/ScrapSort/Recycling/test_rgb888",img)
        return img#,test_img

    # dataset size is number of images
    def __len__(self):
        return len(self.imgs)
    
    # how to get one sample from the dataset
    def __getitem__(self, idx):
        # attempt to load the image at the specified index
        try:
            img = self.read_img(self.imgs[idx])
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)

            # if test_img == True:
            #     # save the img after transform 
            #     np.save("/home/geffen_cooper/ScrapSort/Recycling/transform_rgb888",img)
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int)
            return img, label

        # if the image is invalid, show the exception
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Diaply the results of a forward pass for a random batch of 64 samples
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
                
                # for normal forward pass use this line
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

                # for quantized forward pass use this line
                #print(imgs[idx].size(),torch.min(imgs[idx]))
                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)+1)/2)

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()

    def viz_mispredict(self,wrong_samples,wrong_preds,actual_preds):
        import matplotlib
        matplotlib.use('TkAgg')
        obj_classes = list(self.classes)
        num_samples = len(wrong_samples)
        num_rows = int(np.floor(np.sqrt(num_samples)))

        if num_rows > 0:
            num_cols = num_samples // num_rows
        else:
            return

        fig,ax_array = plt.subplots(num_rows,num_cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i*num_rows+j
                sample = wrong_samples[idx]
                wrong_pred = wrong_preds[idx]
                actual_pred = actual_preds[idx]
                # Undo normalization
                sample = (sample.permute(1, 2, 0)+1)/2
                text = "L: " + obj_classes[actual_pred.item()]  + " P:",obj_classes[wrong_pred.item()]#", i=" +str(idxs[idx].item())
                
                # for normal forward pass use this line
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

                # for quantized forward pass use this line
                #print(imgs[idx].size(),torch.min(imgs[idx]))
                try:
                    if(ax_array.ndim > 1):
                        ax_array[i,j].imshow(sample)

                        ax_array[i,j].title.set_text(text)
                        ax_array[i,j].set_xticks([])
                        ax_array[i,j].set_yticks([])
                except:
                    return
        plt.show()

    
    # visualize high dimensional embedding using T-SNE
    def viz_emb(self, model,device):
        import matplotlib
        matplotlib.use('TkAgg')

        output_emb = torch.zeros((0,64))
        labels = []
        data_loader = DataLoader(self,128,shuffle=True)

        # iterate through the validation set
        for validation_step, (inputs, target) in enumerate(data_loader):
            with torch.no_grad():
                inputs, target = inputs.to(device), target.to(device)
                # compute output from model
                output = model(inputs)
                output_emb = torch.cat((output_emb, output.detach().cpu()), 0)
                labels.extend(target.detach().cpu().tolist())

        out_emb = np.array(output_emb)
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(output_emb)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = 7
        for lab in range(num_categories):
            indices = labels==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig("viz.pdf")
        



'''Function to get the datasets'''
def sorting_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    # location of images, depends on machine
    #img_dir_path = "/home/geffen/Desktop/sorting_dataset/sorting_imgs/"
    img_dir_path = "/home/geffen_cooper/ScrapSort/Recycling/recycling_dataset"
    #img_dir_path = "/home/geffen_cooper/ScrapSort/Recycling/recycling_dataset/labeled"

    # transforms for training
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomResizedCrop(128),
            transforms.ColorJitter(brightness=(0.85,1.15),saturation=(0.85,1.15),contrast=(0.85,1.15),hue=(-0.1,0.1)),
            transforms.RandomGrayscale(0.25),
            transforms.RandomAffine(degrees=180,translate=(0.15,0.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
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


 #---------------------------------------------------------------------------------------------------------

class ContrastiveSortingDataset(Dataset):
    def __init__(self,img_dir_path,transform):
        self.img_dir_path = img_dir_path
        self.transform = transform
        
        # collect img classes from dir names
        #img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        #self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        #print(self.classes)
        
        # get all training samples/labels by getting absolute paths of the images in each subfolder
        self.imgs = [] # absolute img paths (all images)
        #self.labels = [] # integer labels (all labels in corresponding order)

        i = 0 # index into dataset lists

        # iterate through the dataset directory tree
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            # each execution of this inner loop is for each subdirectory
            if idx > 0: # don't include files in the top folder (subfolders are in the next itertion, idx > 0)
                for file in path_obj[2]: # path_obj[2] is list of files in the object class subdirectories
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    #self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get label from directory name
                    i+=1
        print(len(self.imgs))
        print(img_dir_path)

    # since the image are in RGB565 we need a special function to read them into RGB888             
    def read_img(self,img_path):
        # img_test_path = os.path.join(os.path.basename(os.path.dirname(img_path)),os.path.basename(img_path))
        # test_img = False
        # if(img_test_path == "Plastic/img0190"):
        #     test_img = True

        # width, height, channel
        w = 128
        h = 128
        c = 3

        # images are 8 bits per channel
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        # each RGB565 pixel is two bytes
        #            High byte    Low byte
        # RGB565 --> RRRRRGGG     GGGBBBBB
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
            
            # try to read the next pixel
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)

        # if test_img == True:
        #     # save the img after converting to rgb888  
        #     np.save("/home/geffen_cooper/ScrapSort/Recycling/test_rgb888",img)
        return img

    # dataset size is number of images
    def __len__(self):
        return len(self.imgs)
    
    # how to get one sample from the dataset
    def __getitem__(self, idx):
        # attempt to load the image at the specified index
        try:
            img = self.read_img(self.imgs[idx])

            # np.save("orig",img)
            
            # apply any transformation
            if self.transform:
                img1 = self.transform(img)
                img2 = self.transform(img)
                # np.save("t1",img1)
                # np.save("t2",img2)
            
            # return the sample (img (tensor)), object class (int)
            return img1,img2

        # if the image is invalid, show the exception
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Diaply the results of a forward pass for a random batch of 64 samples
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
                
                # for normal forward pass use this line
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

                # for quantized forward pass use this line
                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)+128)/255)

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()

    
    # visualize high dimensional embedding using T-SNE
    def viz_emb(self, model,device):
        import matplotlib
        matplotlib.use('TkAgg')

        output_emb = torch.zeros((0,64))
        labels = []
        data_loader = DataLoader(self,128,shuffle=True)

        # iterate through the validation set
        for validation_step, (inputs, target) in enumerate(data_loader):
            with torch.no_grad():
                inputs, target = inputs.to(device), target.to(device)
                # compute output from model
                output = model(inputs)
                output_emb = torch.cat((output_emb, output.detach().cpu()), 0)
                labels.extend(target.detach().cpu().tolist())

        out_emb = np.array(output_emb)
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(output_emb)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = 7
        for lab in range(num_categories):
            indices = labels==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig("viz.pdf")
        



'''Function to get the datasets'''
def contrastive_sorting_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    # location of images, depends on machine
    #img_dir_path = "/home/geffen/Desktop/sorting_dataset/sorting_imgs/"
    img_dir_path = "/home/geffen_cooper/ScrapSort/Recycling/recycling_dataset/"

    # transforms for training
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomResizedCrop(128),
            transforms.ColorJitter(brightness=(0.85,1.15),saturation=(0.85,1.15),contrast=(0.85,1.15),hue=(-0.1,0.1)),
            transforms.RandomGrayscale(0.25),
            transforms.RandomAffine(degrees=180,translate=(0.15,0.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        test_dataset = None
        
    # create the datasets
    train_dataset = ContrastiveSortingDataset(os.path.join(img_dir_path,"train"),train_transform)
    test_dataset = ContrastiveSortingDataset(os.path.join(img_dir_path,"test"),test_transform)
    
    return train_dataset, test_dataset


'''Function to get the datasets'''
def sorting_get_datasets_downstream(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    # location of images, depends on machine
    #img_dir_path = "/home/geffen/Desktop/sorting_dataset/sorting_imgs/"
    img_dir_path = "/home/geffen_cooper/ScrapSort/Recycling/recycling_dataset/"

    # transforms for training
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomResizedCrop(128),
            transforms.ColorJitter(brightness=(0.85,1.15),saturation=(0.85,1.15),contrast=(0.85,1.15),hue=(-0.1,0.1)),
            transforms.RandomGrayscale(0.25),
            transforms.RandomAffine(degrees=180,translate=(0.15,0.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
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




'''
Sorting Dataset Class for bounding boxes
Parameters:
  img_dir_path - Full path to directory with the images for this dataset.
                 This assumes that the subdirectories contain each class, 
                 only images are in these subdirectories, and that the
                 subdirectory basenames are the desired name of the object class.
                 i.e. dog/dog1.png, cat/cat1.png, etc.

  transform -    Specifies the image format (size, RGB, etc.) and augmentations to use
  normalize -    Specifies whether to make the image zero mean, unit variance

CSV:
  In the top level directory, include the CSVs with the bounding boxes for each class.
  Use the naming convention Class.csv. Also use this format "filename,x,y,w,h".
'''
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
        
        # get all training samples/labels by getting absolute paths of the images in each subfolder
        self.imgs = [] # absolute img paths (all images)
        self.labels = [] # integer labels (all labels in corresponding order)

        i = 0 # index into dataset lists

        # iterate through the dataset directory tree
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            # each execution of this inner loop is for each subdirectory
            if idx > 0: # don't include files in the top folder (subfolders are in the next itertion, idx > 0)
                for file in path_obj[2]: # path_obj[2] is list of files in the object class subdirectories
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get label from directory name
                    i+=1
                    
    def read_img(self,img_path):
        # width, height, channel
        w = 128
        h = 128
        c = 3

        # images are 8 bits per channel
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        # each RGB565 pixel is two bytes
        #            High byte    Low byte
        # RGB565 --> RRRRRGGG     GGGBBBBB
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
            
            # try to read the next pixel
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)
            
        return img

    
     # dataset size is number of images
    def __len__(self):
        return len(self.imgs)
    
    # how to get one sample from the dataset
    def __getitem__(self, idx):
        # attempt to load the image at the specified index
        try:
            # read the image
            img = self.read_img(self.imgs[idx])

            # get the file name based on the idx, append .png because pngs were used for drawing the bounding box
            file_name = os.path.basename(self.imgs[idx])+".png"
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # if class is 'none' then bb doesn't matter
            if label == 2:
                return img,label,torch.tensor([0.01,0.01,0.01,0.01]).float()
            
            # read the csv as a pandas dataframe
            df = pd.read_csv(os.path.dirname(self.img_dir_path) + "/" + list(self.classes)[label]+".csv")
            
            # grab the row based on the file name
            row = df.loc[df['filename'] == file_name]

            # extract the vounding box values as a tensor
            bb = [row['x'].item(),row['y'].item(),row['w'].item(),row['h'].item()]
            bb = torch.tensor(bb).float()
            
            # return the sample (img (tensor)), object class (int),bb coordinates
            return img, label, bb
        
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
        (imgs, labels, bbs) = next(iter(data_loader))
        (imgs, labels, bbs) = next(iter(data_loader))
        
        # batch output
        predictions = model(imgs)
        none_idxs = (labels == 2).nonzero()
        bbs[none_idxs] = 1.
        #div = (predictions[:,6:10].detach()/bbs)
        #print(div.mean(dim=0))
        
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
                
                # get a row in the batch --> [c0, c1, c2, c3, c4, c5, x, y, w, h]
                preds = predictions[idx].detach()
                class_pred = preds[0:4]
                bb_preds = preds[4:8]
                print("p:",bb_preds)
                print("gt:",bbs[idx])
                bb_preds[0] = bb_preds[0]/3500
                bb_preds[1] = bb_preds[1]/3500
                bb_preds[2] = bb_preds[2]/3500
                bb_preds[3] = bb_preds[3]/3500

                # bb_preds[0] = preds[5]/30720
                # bb_preds[1] = preds[6]/30464
                # bb_preds[2] = preds[7]/33792
                # bb_preds[3] = preds[8]/31360
                
                
                text = "P:" + str(obj_classes[class_pred.argmax().item()]) + "  GT:" + str(obj_classes[labels[idx]])
                
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0)+1)
                #ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)+128)/255)
                #print(imgs[idx].permute(1, 2, 0))
                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)+1)/2)
                rect_gt = patches.Rectangle((bbs[idx][0],bbs[idx][1]),bbs[idx][2],bbs[idx][3], edgecolor='r', facecolor="none",lw=1.5)
                rect_pd = patches.Rectangle((bb_preds[0].item(),bb_preds[1].item()),bb_preds[2].item(),bb_preds[3].item(), edgecolor='g', facecolor="none",lw=1.5)
                if labels[idx] < 5:
                    ax_array[i,j].add_patch(rect_gt)
                    ax_array[i,j].add_patch(rect_pd)
                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()
        
        
'''Function to get the datasets'''
def sorting_get_datasetsbb(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    #img_dir_path = "/home/geffen/Desktop/sorting_imgs_all/"
    img_dir_path = "/home/geffen_cooper/ScrapSort/Recycling/recycling_dataset/bb_imgs"

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.ColorJitter(brightness=(0.90,1.10)),#,saturation=(0.5,1),contrast=(0.7,1.1)),
            #transforms.RandomGrayscale(0.05),
            #transforms.RandomAffine(degrees=5,translate=(0.05,0.05)),
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
        'output': ('Paper','Metal','Plastic', 'None'),
        'loader': sorting_get_datasets,
    },
    {
        'name': 'sortingss',
        'input': (3, 128, 128),
        'output': ('Paper','Metal','Plastic', 'None'),
        'loader': contrastive_sorting_get_datasets,
    },
    {
        'name': 'sortingssd',
        'input': (3, 128, 128),
        'output': ('Paper','Metal','Plastic', 'None'),
        'loader': sorting_get_datasets_downstream,
    },
    {
        'name': 'sortingbb',
        'input': (3, 128, 128),
        'output': ('Paper','Metal','Plastic','Other', 'None'),
        'loader': sorting_get_datasetsbb,
    }
]
    
    
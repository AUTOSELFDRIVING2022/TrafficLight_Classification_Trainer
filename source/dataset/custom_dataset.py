import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision
import cv2
import numpy as np
import glob2
import random
import albumentations as A
import albumentations.pytorch as Ap

from torchsampler import ImbalancedDatasetSampler

NUM_FRAMES = 10
#

class SequenceDataset(Dataset):
    def __init__(self, file, interval=1, max_len=10, transform=None, train=True):
        super().__init__()
        self.file = file
        self.len = len(self.file)
        self.interval = interval
        self.max_len = max_len
        self.transform = transform
        self.train = train
        self._class_names = {'Green': 0, 'Green_Left': 1, 'Red': 2, 'Red_Left': 3, 'Red_Yellow': 4, 'Yellow': 5, 'Yellow_Warning': 6}
    
    def __getitem__(self, idx):
        file = self.file[idx]
        imageFolder = sorted(glob2.glob(file + "/*.png"))
        folderName = file.split("/")[-1]
        txtFile = file +  "/" + 'label'
        with open(txtFile, "rb") as f:
            labelTxt = f.readline()  

        label = self._class_names[labelTxt.decode('utf-8').strip()]
        label = torch.as_tensor(label, dtype=torch.long)

        trainImages = []
        if len(imageFolder) > self.max_len:
            start = random.randint(0, len(imageFolder)-1-self.interval*self.max_len)
        else: 
            start = 0

        for i in range(start, start+self.interval*self.max_len):
            if (i - start) % self.interval == 0:
                pil_image = Image.open(imageFolder[i])               
                arr = np.array(pil_image)       
                if self.transform:
                    augmented = self.transform(image=arr) 
                    image = augmented['image']
                
                image = image/255.
                
                trainImages.append(image)
        C, H, W = image.shape
        video = torch.stack(trainImages)
             
        frames = video.permute(0,1,2,3)
        #return frames, label, _
        return {'image': frames, 'label': label, 'name': 'none'}  
        
    def __len__(self):
        return self.len

    def _add_padding(self, video, max_len):
        if video.shape[0] < max_len:
            T, C, H, W = video.shape
            pad = torch.zeros(max_len-T, C, H, W)
            video = torch.cat([video, pad], dim=0)
        else:
            video = video[:max_len]

        return video

class SingleImageDataset(Dataset):
    def __init__(self, data_set_path, transform=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes, self.class_sample_number = self.read_data_set()
        self.transforms = transform
        
    def read_data_set(self):
        all_img_files = []
        all_labels = []

        class_names = sorted(os.walk(self.data_set_path).__next__()[1])

        #_class_names = ['green','green_left','red_left','red','yellow','off','other']
        #_class_names = ['green','green_left','off','other','red','red_left', 'red_yellow', 'yellow', 'yellow21', 'yellow_green4', 'yellow_other', ] #11
        _class_names = ['green','green_left','off','other','pedestrain','red','red_left', 'red_yellow', 'yellow', 'yellow21', 'yellow_green4', 'yellow_other'] #12
        class_sample_number = np.zeros(len(_class_names))
        for index, class_name in enumerate(class_names):
            label = _class_names.index(class_name)
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            
            class_sample_number[label] = len(img_files)
            
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                #img = Image.open(img_file)
                img = cv2.imread(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        return all_img_files, all_labels, len(all_img_files), len(class_names), class_sample_number

    def __getitem__(self, index):
        # image = cv2.imread(self.image_files_path[index])
        # image = cv2.resize(image, (64,32))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.transpose(image,(2,0,1))
        # image = (image / 255.0)
        try:
            if len(index) > 0:
                images = []
                labels = []
                image_paths = []
                for i in range(len(index)):
                    pil_image = Image.open(self.image_files_path[index[i]])               
                    arr = np.array(pil_image)       
                    if self.transforms:
                        augmented = self.transforms(image=arr) 
                        image = augmented['image']
                    
                    image = image/255.
                    images.append(image)
                    labels.append(self.labels[index[i]])
                    image_paths.append(self.image_files_path[index[i]])
                    
                return {'image': images, 'label': labels, 'name': image_paths}         
            else: 
                pil_image = Image.open(self.image_files_path[index])               
                arr = np.array(pil_image)       
                if self.transforms:
                    augmented = self.transforms(image=arr) 
                    image = augmented['image']
                
                image = image/255.
                return {'image': image, 'label': self.labels[index], 'name':self.image_files_path[index]}         
        except:
            pil_image = Image.open(self.image_files_path[index])               
            arr = np.array(pil_image)       
            if self.transforms:
                augmented = self.transforms(image=arr) 
                image = augmented['image']
            
            image = image/255.
            return {'image': image, 'label': torch.tensor(self.labels[index]), 'name':self.image_files_path[index]}         
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        # image_hsv[:,:,0] = (image_hsv[:,:,0] / 180.0) #Range of Hue is 0 to 180 for 1byte in OpenCV
        # image_hsv[:,:,1] = (image_hsv[:,:,1] / 255.0)
        # image_hsv[:,:,2] = (image_hsv[:,:,2] / 255.0)
        # image_hsv = np.transpose(image_hsv,(2,0,1))
        # return {'image': image_hsv, 'label': self.labels[index]}

        # image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # image_lab = np.transpose(image_lab,(2,0,1))
        # image_lab = (image_lab / 255.0)
        # return {'image': image_lab, 'label': self.labels[index], 'name':self.image_files_path[index]}

    def __len__(self):
        return self.length
    
def get_dataset(cfg):
    # albumentations_train = A.Compose([
    #     A.Resize(cfg.train_config.img_size[0], cfg.train_config.img_size[1]), 
    #     A.OneOf([
    #         A.MotionBlur(p=0.5),
    #         A.OpticalDistortion(p=0.5),
    #         A.GaussNoise(p=0.5)                 
    #     ], p=0.5),
    #     A.Blur(p=0.2),
    #     A.CLAHE(p=0.4),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
    #     Ap.transforms.ToTensorV2()])
    albumentations_train = A.Compose([
        A.Resize(cfg.train_config.img_size[0], cfg.train_config.img_size[1]), 
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.GaussNoise(p=0.5),   
            A.JpegCompression(p=0.5),              
        ], p=0.5),
        A.OneOf([
            A.Blur(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=0, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=0, b_shift_limit=0, p=0.5),
        ], p=0.5),
        Ap.transforms.ToTensorV2()])

    albumentations_valid = A.Compose([
        A.Resize(cfg.train_config.img_size[0], cfg.train_config.img_size[1]), 
        Ap.transforms.ToTensorV2()])
        
    if cfg.train_config.model_type == 'single_frame':
        if cfg.dataset.name == 'single':
            train_data_set = SingleImageDataset(data_set_path = cfg.dataset.train_path, transform=albumentations_train)
            
            class_sample_count =  torch.FloatTensor(train_data_set.class_sample_number) # train_set.weights : [296, 3381, 12882, 12857, 1016]
            _weight = 1. / class_sample_count
            #_weight = _weight.view(1, train_data_set.num_classes)
            _weight = _weight.double()

            sampler = torch.utils.data.sampler.WeightedRandomSampler(_weight, cfg.train_config.train_bs)

            train_loader = DataLoader(train_data_set, batch_size = cfg.train_config.train_bs, shuffle=True)
            #train_loader = DataLoader(train_data_set, batch_size = cfg.train_config.train_bs, sampler=sampler)
            #train_loader = DataLoader(train_data_set, batch_size = cfg.train_config.train_bs, sampler=ImbalancedDatasetSampler(train_data_set))

            test_data_set = SingleImageDataset(data_set_path = cfg.dataset.valid_path, transform=albumentations_valid)
            test_loader = DataLoader(test_data_set, batch_size = cfg.train_config.valid_bs, shuffle=False)
            return train_loader, test_loader
        elif cfg.dataset.name == 'single_image_folder':
            # Data loading code
            traindir = cfg.dataset.train_path
            valdir = cfg.dataset.valid_path
            normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize((64, 128)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))
            if cfg.dataset.use_sampler:
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=cfg.train_config.train_bs, 
                    shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))    
            else: 
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=cfg.train_config.train_bs, 
                    shuffle=True, sampler=None)

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(
                    valdir, torchvision.transforms.Compose([
                        torchvision.transforms.Resize((64, 128)),
                        #torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.ToTensor(),
                        normalize,
                ])), batch_size=cfg.train_config.valid_bs, 
                shuffle=False)
            return train_loader, test_loader
    elif cfg.train_config.model_type == 'temporal_frame':
        ### Define train and test loader
        trainVideoFolder = sorted(glob2.glob(cfg.dataset.train_path + "*"))
        validVideoFolder = sorted(glob2.glob(cfg.dataset.valid_path + "*"))

        trainVideo = []
        validVideo = []
        for i in range(len(trainVideoFolder)):
            trainVideo.append(trainVideoFolder[i])

        for i in range(len(validVideoFolder)):
            validVideo.append(validVideoFolder[i])

        trainDataset = SequenceDataset(trainVideo, max_len = cfg.dataset.num_frames, transform=albumentations_train)
        validDataset = SequenceDataset(validVideo, max_len = cfg.dataset.num_frames, transform=albumentations_valid)
        trainLoader = DataLoader(trainDataset, batch_size=cfg.train_config.train_bs, num_workers=cfg.dataset.num_workers, shuffle=True)
        validLoader = DataLoader(validDataset, batch_size=cfg.train_config.valid_bs, num_workers=cfg.dataset.num_workers, shuffle=False)
        return trainLoader, validLoader
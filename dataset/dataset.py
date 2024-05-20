import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import  image_train,image_test

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class video_BP4D_train(Dataset):
    def __init__(self, root_path, length=16, fold=1, transform=None, crop_size=224, loader=default_loader):
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.length = length
        self.fold = fold
        self.img_folder_path = os.path.join(self._root_path, 'img')
        self.data = []
        # img
        train_image_list_path = os.path.join(self._root_path, 'BP4D_train_img_path_fold'+str(self.fold)+'.txt')
        train_image_list = open(train_image_list_path).readlines()

        # img labels
        train_label_list_path = os.path.join(self._root_path, 'BP4D_train_label_fold'+str(self.fold)+'.txt')
        train_label_list = np.loadtxt(train_label_list_path)

        #region labels
        train_sub_label_list_path = os.path.join(self._root_path, 'BP4D_train_sub_label_fold'+str(self.fold)+'.txt')
        train_sub_label_list = np.loadtxt(train_sub_label_list_path)

        video_dic = {}   #   each subject has 8 video with different numbers of frame
        img_path = []
        img_label = []
        img_sub_label = []
        frame_num = 0
        for i in range(len(train_label_list)):
            img_path.append(train_image_list[i])
            img_label.append(train_label_list[i, :])
            img_sub_label.append(train_sub_label_list[i, :])
            frame_num += 1

            if i+1 < len(train_label_list):
                if train_image_list[i].split('/')[0] != train_image_list[i+1].split('/')[0] or train_image_list[i].split('/')[1] != train_image_list[i+1].split('/')[1]:  #split
                    video_dic['img_path'] = img_path
                    video_dic['img_label'] = np.array(img_label)
                    video_dic['img_sub_label'] = np.array(img_sub_label)
                    video_dic['frame_num'] = frame_num
                    self.data.append(video_dic)
                    video_dic = {}
                    img_path = []
                    img_label = []
                    img_sub_label = []
                    frame_num = 0
            else:
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)
                video_dic['img_sub_label'] = np.array(img_sub_label)
                video_dic['frame_num'] = frame_num
                self.data.append(video_dic)

    def __getitem__(self, index):
            video_dic = self.data[index]
            img_path_all, img_label_all, img_sub_label_all, frame_num = video_dic['img_path'], video_dic['img_label'], video_dic['img_sub_label'], video_dic['frame_num']

            random_start_frame = random.randint(0, frame_num-self.length)

            img_path = img_path_all[random_start_frame:random_start_frame+self.length]
            img_label = img_label_all[random_start_frame:random_start_frame+self.length, :]
            img_sub_label = img_sub_label_all[random_start_frame:random_start_frame+self.length, :]

            video = []

            w, h = 256, 256
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.6, 1.4)

            for i in range(len(img_path)):
                img = self.loader(os.path.join(self.img_folder_path, img_path[i]).strip())
                img = self._transform(img, offset_x, offset_y, flip, brightness, contrast, saturation)
                video.append(img)

            video = torch.stack(video)
            return video, img_label, img_sub_label

    def __len__(self):
        return len(self.data)


class video_BP4D_val(Dataset):
    def __init__(self, root_path, length=16, fold=1, transform=None, crop_size=224, loader=default_loader):
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.length = length
        self.fold = fold
        self.img_folder_path = os.path.join(self._root_path, 'img')

        self.data = []
        # img
        val_image_list_path = os.path.join(self._root_path, 'BP4D_test_img_path_fold'+str(self.fold)+'.txt')
        val_image_list = open(val_image_list_path).readlines()

        # img labels
        val_label_list_path = os.path.join(self._root_path, 'BP4D_test_label_fold'+str(self.fold)+'.txt')
        val_label_list = np.loadtxt(val_label_list_path)

        val_sub_label_list_path = os.path.join(self._root_path, 'BP4D_test_sub_label_fold'+str(self.fold)+'.txt')
        val_sub_label_list = np.loadtxt(val_sub_label_list_path)

        video_dic = {}    # split all frames to clips with fixed length
        img_path = []
        img_label = []
        img_sub_label = []
        frame_num = 0
        for i in range(len(val_label_list)):
            img_path.append(val_image_list[i])
            img_label.append(val_label_list[i, :])
            img_sub_label.append(val_sub_label_list[i, :])
            frame_num += 1
            if frame_num == self.length:
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)
                video_dic['img_sub_label'] = np.array(img_sub_label)
                self.data.append(video_dic)
                video_dic = {}
                img_path = []
                img_label = []
                img_sub_label = []
                frame_num = 0

            elif i+1 < len(val_label_list):
                if val_image_list[i].split('/')[1] != val_image_list[i+1].split('/')[1] or int(val_image_list[i].split('/')[-1].split('.')[0]) + 1 != int(val_image_list[i+1].split('/')[-1].split('.')[0]):
                    while frame_num < self.length:
                        img_path.append(img_path[-1])        #      padding with the last image
                        img_label.append(-np.ones(12))       #      label padding with -1
                        img_sub_label.append(img_sub_label[-1])   # sub_label padding with the last sub_label
                        frame_num += 1
                    video_dic['img_path'] = img_path
                    video_dic['img_label'] = np.array(img_label)
                    video_dic['img_sub_label'] = np.array(img_sub_label)

                    self.data.append(video_dic)
                    video_dic = {}
                    img_path = []
                    img_label = []
                    img_sub_label = []
                    frame_num = 0
            else:
                while frame_num < self.length:
                    img_path.append(img_path[-1])
                    img_label.append((-1) * np.ones(12))
                    img_sub_label.append(img_sub_label[-1])
                    frame_num += 1
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)
                video_dic['img_sub_label'] = np.array(img_sub_label)

                self.data.append(video_dic)

    def __getitem__(self, index):
            video_dic = self.data[index]
            img_path, img_label,img_sub_label = video_dic['img_path'],video_dic['img_label'],video_dic['img_sub_label']

            video = []

            for i in range(len(img_path)):

                img = self.loader(os.path.join(self.img_folder_path, img_path[i]).strip())
                img = self._transform(img)
                video.append(img)

            video = torch.stack(video)

            return video, img_label, img_sub_label

    def __len__(self):
        return len(self.data)

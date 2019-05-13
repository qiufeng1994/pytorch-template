import scipy.io
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import glob
import scipy.io
import random


class Huawei_Data(Dataset):
    def __init__(self, root_path, list_path, is_training, transform=None, data_format='png'):
        self.list_path = list_path
        self.root_path = root_path
        self.transform = transform
        self.is_training = is_training
        self.data_format = data_format
        with open(self.list_path, 'r') as f:
            list_old = f.readlines()
        self.list = []
        for path in list_old:
            imgs_path = glob.glob(os.path.join(self.root_path, path.split(',')[0], '*.' + self.data_format))
            num_frames = len(imgs_path)
            if num_frames >= 8:
                self.list.append(path)
    
    def _get_val_indices(self, num_frames):
        if num_frames >= 16:
            sample = np.arange(0, 16, 2)
            res = num_frames - 16
            offset = np.random.randint(0, res + 1)
            offsets = sample + offset
        elif num_frames >= 8:
            sample = np.arange(0, 8, 1)
            res = num_frames - 8
            offset = np.random.randint(0, res + 1)
            offsets = sample + offset
        else:
            offsets = np.zeros((8))
        return offsets
    
    def _get_sequence(self, num_frames):
        
        if num_frames < 16:
            window = 8
        elif num_frames < 32:
            window = 16
        else:
            window = 32
        sample = np.arange(0, window, window // 8)
        res = num_frames - window
        try:
            offset = np.random.randint(0, res + 1)
        except:
            offset = 0
        offsets = sample + offset
        return offsets
    
    def _get_sequence_glob(self, num_frames):
        
        offset = np.arange(0, 8 * (num_frames // 8), num_frames // 8)
        res = np.random.randint(0, 1 + num_frames % 8)
        return offset + res
    
    def _get_val_indices_random(self, num_frames):
        
        offsets = np.random.choice(num_frames, 8, replace=False)
        offsets.sort()
        return offsets
    
    def _load_image(self, img_path):
        
        img = Image.open(img_path).convert('RGB')
        
        return img
    
    def _load_mat_2_img(self, mat_path):
        
        mat = scipy.io.loadmat(mat_path)['resized_rgb']
        img = Image.fromarray(mat)
        img = img.resize((640, 480))
        return img
    
    def _new_label(self, label):
        if label == 0:
            label = 1
        elif label == 1:
            label = 2
        elif label == 2:
            label = 3
        elif label == 4:
            label = 4
        elif label == 5:
            label = 5
        elif label == 6:
            label = 6
        elif label == 7:
            label = 7
        elif label == 10:
            label = 8
        else:
            label = 0
        return label
    
    def _take_num(self, path):
        return int(path.split('/')[-1].split('.')[0])
    
    def _random_flip(self, imgs, label):
        
        if random.random() < 0.5:
            label = int(label)
            imgs_flip = []
            for img in imgs:
                img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                imgs_flip.append(img_flip)
            if label == 1:
                label_flip = 2
            elif label == 2:
                label_flip = 1
            elif label == 4:
                label_flip = 5
            elif label == 5:
                label_flip = 4
            else:
                label_flip = label
            return imgs_flip, label_flip
        else:
            return imgs, label
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        imgs_path = glob.glob(os.path.join(self.root_path, self.list[idx].split(',')[0], '*.' + self.data_format))
        label = self.list[idx].split(',')[-1].strip('\n')
        # label = self._new_label(int(label))
        imgs_path.sort(key=self._take_num)
        num_frames = len(imgs_path)
        if self.is_training:
            indices = self._get_sequence(num_frames)
        else:
            indices = self._get_sequence(num_frames)
        images = []
        for seg_ind in indices:  # indices: [2,4,7,9,12,14,17,19]
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(imgs_path[p])
            except:
                print('lenth {}, index {}, file{}'.format(num_frames, indices, self.list[idx]))
                seg_imgs = self._load_image(imgs_path[int(indices[3])])
                with open('jpg_error_img.txt', 'a+') as f:
                    f.write(imgs_path + '\n')
            images.append(seg_imgs)
        
        images, label = self._random_flip(images, label)
        
        if self.transform:
            images = self.transform(images)
        # return images, int(label_flip), self.list[idx].split(' ')[0]
        return images, int(label)


class pipelineDataset(Dataset):
    def __init__(self, root_path, img_dir, transform=None):
        img_path = os.path.join(root_path, img_dir)
        self.imgs = glob.glob(img_path + '*.png')
        self.imgs.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs) // 8
    
    def __getitem__(self, idx):
        imgs_seg = self.imgs[idx * 8: (idx + 1) * 8]
        imgs_seg.sort()
        images = [Image.open(i).convert('RGB') for i in imgs_seg]
        if self.transform:
            images = self.transform(images)
        return images, imgs_seg


if __name__ == '__main__':
    mat = scipy.io.loadmat(
        '/Data/Data3/Data_new/GestureData2/20181106_3#/201811060958009_02/201811060958009_02_clean/frame_48.mat')[
        'resized_rgb']
    print(1)
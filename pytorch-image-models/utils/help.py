import os
import re
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from timm.data import ImageDataset, IterableImageDataset, AugMixDataset, create_loader


class LandmarkDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        # print(image_filepath)
        image = cv2.imread(image_filepath)
        # print(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = image_filepath.split('/')[-2]
        # label = class_to_idx[label]
        label = re.match(r'.*/n\d*_(\d)\..*', image_filepath).group(1)
        # if self.transform is not False:
        #     image = self.transform(image=image)["image"]    
        return image, label
    

def create_dataset_historical(path, split, is_training=False, batch_size=None, **kwargs):
    if split == 'train':
        path = os.path.join(path, 'HistFigsClass8-rgb-train')
    elif split == 'validation':
        path = os.path.join(path, 'HistFigsClass8-rgb-eval')
    else:
        raise ValueError(f'Unknown split: {split}')
    
    image_paths = []
    # classes = []
    for img_path in os.listdir(path):
        # classes.append(re.match(r'n\d*_(\d)\..*', img_path).group(1)) 
        image_paths.append(os.path.join(path, img_path))

    # dataset = LandmarkDataset(image_paths)
    ds = ImageDataset(path, parser='pil', **kwargs)
    # data_loader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True
    # )
    return ds

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
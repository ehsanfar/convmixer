import os
import cv2
import re
from torch.utils.data import Dataset, DataLoader

from .dataset import IterableImageDataset, ImageDataset


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds


def create_dataset_historical(path, split, is_training=False, batch_size=None, **kwargs):
    if split == 'train':
        root = os.path.join(path, 'HistFigsClass8-rgb-train')
    elif split == 'validation':
        root = os.path.join(path, 'HistFigsClass8-rgb-val')
    else:
        raise ValueError(f'Unknown split: {split}')
    
    image_paths = []
    classes = []
    for img_path in os.listdir(path):
        classes.append(re.match(r'n\d*_(\d)\..*', img_path).group(1)) 
        image_paths.append(os.path.join(path, img_path))

    dataset = LandmarkDataset(image_paths)
    # ds = ImageDataset(root, parser='pil', **kwargs)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # label = image_filepath.split('/')[-2]
        # label = class_to_idx[label]
        label = re.match(r'.*/n\d*_(\d)\..*', image_filepath).group(1)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    

import torch
import json
import os
import sys
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """
    A PyTorch dataset class to be used in the PyTorch DataLoader to create batches.
    """
    def __init__(self, image_dir, transform = None):
        """
        Args:
            image_dir (string) : Directory with all the train images
            TODO : include all the training data at one place.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = [img for img in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, img))]

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_name)
        if self.transform is not None:
            image = self.transform(image)
        return image


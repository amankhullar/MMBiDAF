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
    def __init__(self, images_dir, transform = None):
        """
        Args:
            images_dir (string) : Directory with all the train images
                                  for all the vidoes
            TODO : include all the training data (all modalities) at one place
        """
        self.images_dir = images_dir
        self.transform = transform
        self.images = self.load_images()
    
    def load_images(self):
        images = []
        for video_path in os.listdir(self.images_dir):
            keyframes = [os.path.join(video_path, img) for img in os.listdir(os.path.join(self.images_dir, video_path)) \
                         if os.path.isfile(os.path.join(self.images_dir, video_path, img))]
            images.extend(keyframes)
        return images

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        image_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(image_name)
        if self.transform is not None:
            image = self.transform(image)
        return image


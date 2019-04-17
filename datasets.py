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
        self.load_images()
    
    def load_images(self):
        for video_path in os.listdir(self.images_dir):
            keyframes = [img for img in os.listdir(os.path.join(self.image_dir, video_path) \
                         if os.path.isfile(os.path.join(self.image_dir, video_path, img))]
            self.images.extend(keyframes)

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_name)
        if self.transform is not None:
            image = self.transform(image)
        return image


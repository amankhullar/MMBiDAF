import torch
import json
import os
import re
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
        self.sorted_image_dir = sorted(os.listdir(self.images_dir), key = int)
        self.images = self.load_images()

    def get_num(self, str):
        return int(re.search(r'\d+', re.search(r'_\d+', str).group()).group())
    
    def load_images(self):
        images = []
        for video_path in self.sorted_image_dir:
            keyframes = [os.path.join(video_path, img) for img in os.listdir(os.path.join(self.images_dir, video_path)) \
                         if os.path.isfile(os.path.join(self.images_dir, video_path, img))]
            keyframes.sort(key = self.get_num)
            images.extend([keyframes])
        return images

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        transformed_images = []
        for image_name in self.images[idx]:
            image_path = os.path.join(self.images_dir, image_name)
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            transformed_images.append(image)
        return transformed_images


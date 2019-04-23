import json
import os
import re
import sys

import numpy as np
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    A PyTorch dataset class to be used in the PyTorch DataLoader to create batches.
    """
    def __init__(self, images_dir, transform = None):
        """
        Args:
            images_dir (string) : Directory with all the train images
                                  for all the videos
            transform (torchvision.transforms.transforms.Compose) : The required transformation required to normalize all images
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
        return torch.stack(transformed_images)

class AudioDataset(Dataset):
    """
    A PyTorch dataset class to be used in the PyTorch DataLoader to create batches of the Audio.
    """
    def __init__(self, audio_dir):
        """
        Args:
            audio_dir (String) : Director containing the MFCC features for all the
                                 audio in a single course
        """
        self.audio_dir = audio_dir
        self.audios = sorted(os.listdir(self.audio_dir), key = self.get_num)

    def get_num(self, str):
        return int(re.search(r'\d+',str).group())

    def __len__(self):
        return len(self.audios)
    
    def __get_item__(self, idx):
        with open(os.path.join(self.audio_dir, self.audios[idx]), 'rb') as fp:
            audio_vectors = pickle.load(fp)
        audio_vectors = np.transpose(audio_vectors)
        audio_vectors = torch.from_numpy(audio_vectors)
        return audio_vectors

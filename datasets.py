import json
import os
import pickle
import re
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    A Pytorch dataset class to be used in the Pytorch Dataloader to create text batches
    """
    def __init__(self, courses_dir):
        """
        Args :
             courses_dir (string) : The directory containing the embeddings for the preprocessed sentences 
        """
        self.courses_dir = courses_dir
        self.text_embeddings_path = self.load_sentence_embeddings_path()

    def load_sentence_embeddings_path(self):
        transcript_embeddings = []

        # Get sorted list of all courses (excluding any files)
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)
        
        for course_number in sorted(dirlist, key=int):
            course_transcript_path = os.path.join(self.courses_dir, course_number, 'sentence_features/')
            text_embedding_path = [self.courses_dir + course_number + '/sentence_features/' + transcript_path for transcript_path in sorted(os.listdir(course_transcript_path), key=self.get_num)]
            transcript_embeddings.append(text_embedding_path)

        return [val for sublist in transcript_embeddings for val in sublist]    #Flatten the list of lists

    def get_num(self, str):
        return int(re.search(r'\d+', str).group())

    def __len__(self):
        return len(self.text_embeddings_path)
    
    def __getitem__(self, idx):
        self.embedding_path = self.text_embeddings_path[idx]
        self.embedding_dict = torch.load(self.embedding_path)
        word_vectors = torch.zeros(len(self.embedding_dict),300)
        for count, sentence in enumerate(self.embedding_dict):
            word_vectors[count] = self.embedding_dict[sentence]
        return word_vectors

class ImageDataset(Dataset):
    """
    A PyTorch dataset class to be used in the PyTorch DataLoader to create batches.

    Member variables:
    self.image_paths (2D list) : A 2D list containing image paths of all the videos.
                                 The first index represents the video, and the
                                 second index represents the keyframe.
    self.num_videos (int) : The total number of videos across courses in the dataset.

    """
    def __init__(self, courses_dir, transform = None):
        """
        Args:
            courses_dir (string) : Directory with all the courses
            transform (torchvision.transforms.transforms.Compose) : The required transformation required to normalize all images
        """
        self.courses_dir = courses_dir
        self.transform = transform
        self.num_videos = 0
        self.image_paths = self.load_image_paths()

    def get_num(self, str):
        return int(re.search(r'\d+', re.search(r'_\d+', str).group()).group())
    
    def load_image_paths(self):
        images = []

        # Get sorted list of all courses (excluding any files)
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)

        for course_dir in sorted(dirlist, key=int):
            keyframes_dir_path = os.path.join(self.courses_dir, course_dir, 'video_key_frames/')
            for video_dir in sorted(os.listdir(keyframes_dir_path), key=int):
                self.num_videos += 1
                video_dir_path = os.path.join(keyframes_dir_path, video_dir)
                keyframes = [os.path.join(video_dir_path, img) for img in os.listdir(video_dir_path) \
                            if os.path.isfile(os.path.join(video_dir_path, img))]
                keyframes.sort(key = self.get_num)
                images.extend([keyframes])

        return images

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        transformed_images = []
        for image_path in self.image_paths[idx]:
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            transformed_images.append(image)
        return torch.stack(transformed_images)

class AudioDataset(Dataset):
    """
    A PyTorch dataset class to be used in the PyTorch DataLoader to create batches of the Audio.
    """
    def __init__(self, courses_dir):
        """
        Args:
            courses_dir (String) : Director containing the MFCC features for all the
                                 audio in a single course
        """
        self.courses_dir = courses_dir
        # self.audios_paths = sorted(os.listdir(self.courses_dir), key = self.get_num)
        self.audios_paths = self.load_audio_path()

    def load_audio_path(self):
        audio_embeddings = []

        # Get sorted list of all courses (excluding any files)
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)
        
        for course_number in sorted(dirlist, key=int):
            course_audio_path = os.path.join(self.courses_dir, course_number, 'audio-features/')
            audio_embedding_path = [self.courses_dir + course_number + '/audio-features/' + audio_path for audio_path in sorted(os.listdir(course_audio_path), key=self.get_num)]
            audio_embeddings.append(audio_embedding_path)

        return [val for sublist in audio_embeddings for val in sublist]     #Flatten the list of lists

    def get_num(self, str):
        return int(re.search(r'\d+', str).group())

    def __len__(self):
        return len(self.audios_paths)
    
    def __getitem__(self, idx):
        with open(self.audios_paths[idx], 'rb') as fp:
            audio_vectors = pickle.load(fp)
        audio_vectors = np.transpose(audio_vectors)
        audio_vectors = torch.from_numpy(audio_vectors)
        return audio_vectors

class TagetDataset(Dataset):
    """
    A Pytorch dataset class to be used in loading target datatset for training and evaluation purpose.
    """
    def __init__(self, courses_dir):
        """
        Args :
             courses_dir (string) : The directory containing the entire dataset.
        """
        self.courses_dir = courses_dir
        self.target_sentences_path = self.load_sentences_path()

        def load_sentences_path(self):
            target_sentences = []
            dirlist = []
            for fname in os.listdir(self.courses_dir):
                if os.path.isdir(os.path.join(self.courses_dir, fname)):
                    dirlist.append(fname)

            for course_number in sorted(dirlist, key=int):
                target_path = os.path.join(self.courses_dir, course_number, 'ground-truth/')
                target_sentence_path = [target_path + target_sentence for target_sentence in sorted(os.listdir(target_path), key=self.get_num)]
                target_sentences.append(target_sentence_path)

            return [val for sublist in target_sentences for val in sublist]    #Flatten the list of lists

        def get_num(self, str):
            return int(re.search(r'\d+', str).group())
        
        def __len__():
            return len(self.target_sentences_path)

        def __get_item__(self, idx):
            self.

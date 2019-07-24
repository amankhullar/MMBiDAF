import json
import os
import pickle
import re
import sys
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize


class TextDataset(Dataset):
    """
    A Pytorch dataset class to be used in the Pytorch Dataloader to create text batches
    """
    def __init__(self, courses_dir, max_text_length=405):
        """
        Args :
             courses_dir (string) : The directory containing the embeddings for the preprocessed sentences 
        """
        self.courses_dir = courses_dir
        with open('dataset_inter.pkl', 'rb') as f:
            self.dataset_inter = pickle.load(f)
        self.text_embeddings_path = self.load_sentence_embeddings_path()
        self.max_text_length = max_text_length

    def load_sentence_embeddings_path(self):
        transcript_embeddings = []

        # Get sorted list of all courses (excluding any files)
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)
        
        for course_number in sorted(dirlist, key=int):
            course_transcript_path = os.path.join(self.courses_dir, course_number, 'sentence_features/')
            for transcript_path in sorted(os.listdir(course_transcript_path), key=self.get_num):
                if '_' in transcript_path:
                    continue
                # Dataset cleanup for consistency
                path_check = course_number + '/' + transcript_path[:-3]
                if path_check not in self.dataset_inter:
                    continue

                path = self.courses_dir + course_number + '/sentence_features/' + transcript_path
                transcript_embeddings.append(path)
        
        return transcript_embeddings

    def get_num(self, str):
        return int(re.search(r'\d+', str).group())

    def __len__(self):
        return len(self.text_embeddings_path)
    
    def __getitem__(self, idx):
        self.embedding_path = self.text_embeddings_path[idx]
        self.embedding_dict = torch.load(self.embedding_path)
        word_vectors = torch.zeros(len(self.embedding_dict)+1, 300)
        for count, sentence in enumerate(self.embedding_dict):
            word_vectors[count] = self.embedding_dict[sentence]
        word_vectors[len(self.embedding_dict)] = torch.zeros(1, 300) - 1            # End of summary token embedding
        return word_vectors, len(self.embedding_dict) + 1                           # Added EOS to the original data

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
        with open('dataset_inter.pkl', 'rb') as f:
            self.dataset_inter = pickle.load(f)
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
                # Dataset cleanup for consistency
                path_check = course_dir + '/' + str(video_dir)
                if path_check not in self.dataset_inter:
                    continue
                
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
        with open('dataset_inter.pkl', 'rb') as f:
            self.dataset_inter = pickle.load(f)
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
            
            for audio_path in sorted(os.listdir(course_audio_path), key=self.get_num):
                # Dataset cleanup for consistency
                path_check = course_number + '/' + audio_path[:-4]
                if path_check not in self.dataset_inter:
                    continue

                path = self.courses_dir + course_number + '/audio-features/' + audio_path
                audio_embeddings.append(path)

        return audio_embeddings

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

class TargetDataset(Dataset):
    """
    A Pytorch dataset class to be used in loading target datatset for training and evaluation purpose.
    """
    def __init__(self, courses_dir):
        """
        Args :
             courses_dir (string) : The directory containing the entire dataset.
        """
        self.courses_dir = courses_dir
        with open('dataset_inter.pkl', 'rb') as f:
            self.dataset_inter = pickle.load(f)
        self.target_sentences_path = self.load_target_sentences_path()
        self.source_sentences_path = self.load_source_sentences_path()

    def load_target_sentences_path(self):
        target_sentences = []
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)

        for course_number in sorted(dirlist, key=int):
            target_path = os.path.join(self.courses_dir, course_number, 'ground-truth/')
            target_sentence_path = [target_path + target_sentence for target_sentence in sorted([item for item in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, item)) and '.txt' in item and '_' not in item and '{}/{}'.format(course_number, item[:-4]) in self.dataset_inter], key=self.get_num)]
            target_sentences.extend(target_sentence_path)

        return target_sentences

    def load_source_sentences_path(self):
        source_sentences = []

        # Get sorted list of all courses (excluding any files)
        dirlist = []
        for fname in os.listdir(self.courses_dir):
            if os.path.isdir(os.path.join(self.courses_dir, fname)):
                dirlist.append(fname)
        
        for course_number in sorted(dirlist, key=int):
            source_path = os.path.join(self.courses_dir, course_number, 'transcripts/')
            source_sentence_path = [source_path + transcript_path for transcript_path in sorted([item for item in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, item)) and '.txt' in item and '{}/{}'.format(course_number, item[:-4]) in self.dataset_inter], key=self.get_num)]

            source_sentences.extend(source_sentence_path)

        return source_sentences

    def get_num(self, str):
        return int(re.search(r'\d+', str).group())
    
    def __len__(self):
        return len(self.target_sentences_path)

    def __getitem__(self, idx):
        lines = []
        try:
            with open(self.source_sentences_path[idx]) as f:
                for line in f:
                    if re.match(r'\d+:\d+', line) is None:
                        line = line.replace('[MUSIC]', '')
                        lines.append(line.strip())
        except Exception as e:
            logging.error('Unable to open file. Exception: ' + str(e))
        else:
            source_text = ' '.join(lines)
        
        source_text = source_text.lower()
        source_sentences = sent_tokenize(source_text)

        lines = []
        try:
            with open(self.target_sentences_path[idx]) as f:
                for line in f:
                    if re.match(r'\d+:\d+', line) is None:
                        line = line.replace('[MUSIC]', '')
                        lines.append(line.strip())
        except Exception as e:
            logging.error('Unable to open file. Exception: ' + str(e))
        else:
            target_text = ' '.join(lines)

        # target_text = target_text.lower()
        target_sentences = sent_tokenize(target_text)
        for idx2 in range(len(target_sentences)):
            target_sentences[idx2] = target_sentences[idx2].lower()

        target_indices = []
        for target_sentence in target_sentences:
            # target_indices.append(torch.Tensor([source_sentences.index(target_sentence)]))
            try:
                target_indices.append(torch.Tensor([self.get_index(source_sentences, target_sentence)]))
            except Exception as e:
                if False:
                    print("Exception: " + str(e))
                    print(self.target_sentences_path[idx])
                    print(target_sentence)
                    print('\n\n--------------------\n\n')
                    print(source_sentences)
                    print('\n-----------------------\n')
                continue
        target_indices.append(torch.Tensor([len(source_sentences)]))                        # Appended the EOS token
        
        return torch.stack(target_indices), self.source_sentences_path[idx], self.target_sentences_path[idx]

    def get_index(self, source_sentences, target_sentence):
        for idx, sent in enumerate(source_sentences):
            if target_sentence in sent:
                return idx

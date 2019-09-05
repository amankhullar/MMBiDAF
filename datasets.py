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
from nltk.corpus import stopwords, words
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer

final_indices_path = 'dataset_inter2.pkl'

class MultimodalDataset(Dataset):
    """
    A Pytorch dataset class to be used in the Pytorch Dataloader to create text batches
    """
    def __init__(self, train_text_path, gt_path, word_idx_path, vid_feat_path, aud_feat_path, max_text_length=1865):
        """
        Args :
             courses_dir (string) : The directory containing the embeddings for the preprocessed sentences 
        """
        self.vid_feat_path = vid_feat_path
        self.aud_feat_path = aud_feat_path

        try:
            with open(train_text_path, 'r') as f:
                transcripts = f.readlines()                 # list of all transcripts
        except Exception as e:
            print("Could not open source transcripts with error : " + str(e))
            sys.exit()

        try:
            with open(gt_path, 'r') as f:
                gt = f.readlines()                          # list of all ground truths
        except Exception as e:
            print("Could not open target summary with error : " + str(e))
            sys.exit()

        try:
            with open(word_idx_path, 'rb') as f:
                word_idx = pickle.load(f)                   # dictionary containing indices of words
        except Exception as e:
            print("Could not open Word_idx dictionary with error : " + str(e))
            sys.exit()

        self.filename_map = []                          # maps the idx to filename
        self.text_idxs = {}          # dictionary containing the indices for tokens in training transcripts
        self.gt_idxs = {}            # dictionary containing the indices for tokens in ground truth summary

        self.create_seq(transcripts, word_idx)      # create seq for source transcripts
        self.create_seq(gt, word_idx, tgt=True)     # create seq for target summaries
          
        self.max_text_length = max_text_length

    def create_seq(self, text, word_idx, tgt=False):
        for trans in text:
            trans = trans.split(' ')
            filename = trans[0]
            if not tgt:
                self.filename_map.append(filename)
            trans = trans[1:]
            idxs = []
            idxs.append(word_idx['<START>'])
            for word in trans:
                if word in word_idx:
                    idxs.append(word_idx[word])
                else:
                    idxs.append(word_idx['<OOV>'])
            idxs.append(word_idx['<END>'])
            if not tgt:
                self.text_idxs[filename] = idxs
            else:
                self.gt_idxs[filename] = idxs

    def __len__(self):
        return len(self.filename_map)
    
    def __getitem__(self, idx):
        filename = self.filename_map[idx]

        src_seq = self.text_idxs[filename]
        tgt_seq = self.gt_idxs[filename]
        src_seq = torch.Tensor(src_seq)         # (src_seq_len)
        tgt_seq = torch.Tensor(tgt_seq)         # (tgt_seq_len)

        # Get the video features
        try:
            vid_feats = np.load(os.path.join(self.vid_feat_path, filename+'.npy'))          # (vid_seq_len, img_feat_size)
        except Exception as e:
            print("Could not find video features : " + str(e))
            sys.exit()
        
        # TODO : Get the Audio features
            aud_feats = torch.Tensor()

        return src_seq, vid_feats, vid_feats, tgt_seq       # TODO : change the second video features to audio features


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
        with open(final_indices_path, 'rb') as f:
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
        return torch.stack(transformed_images), len(transformed_images)

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
        with open(final_indices_path, 'rb') as f:
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

                path = self.courses_dir + course_number + '/audio-features8/' + audio_path
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
        return audio_vectors, int(audio_vectors.size(0))

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
        with open(final_indices_path, 'rb') as f:
            self.dataset_inter = pickle.load(f)
        self.target_sentences_path = self.load_target_sentences_path()
        self.source_sentences_path = self.load_source_sentences_path()
        with open('words_set.pkl', 'rb') as f:
            self.words_set = pickle.load(f)
        self.lemmatizer = WordNetLemmatizer()

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
        text_dataset = TextDataset(self.courses_dir)
        return text_dataset.text_embedding_paths

    def get_num(self, str):
        return int(re.search(r'\d+', str).group())
    
    def __len__(self):
        return len(self.target_sentences_path)

    def __getitem__(self, idx):
        lines = []
        try:
            emb = torch.load(self.source_sentences_path[idx])
        except Exception as e:
            logging.error('Unable to open file. Exception: ' + str(e))
        else:
            source_sentences = emb.keys()

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
        stop_words = stopwords.words('english')
        tweet_tokenizer = TweetTokenizer()
        target_sentences_processed = []
        for idx2 in range(len(target_sentences)):
            target_sentences[idx2] = target_sentences[idx2].lower()
            words = tweet_tokenizer.tokenize(target_sentences[idx2])
            sent = [word for word in words if word not in stop_words]
            if not self.is_blank_sentence(sent): # Ignore blank sentences
                target_sentences_processed.append(' '.join(sent))

        target_indices = []
        for tidx, target_sentence in enumerate(target_sentences_processed):
            # target_indices.append(torch.Tensor([source_sentences.index(target_sentence)]))
            try:
                target_indices.append(torch.Tensor([self.get_index(source_sentences, target_sentence)]))
            except Exception as e:
                if False:
                    print("Exception: " + str(e))
                    print(self.target_sentences_path[idx])
                    print(target_sentences)
                    print('\n--------------------')
                    print(target_sentence)
                    print('\n--------------------')
                    print(target_sentences[tidx])
                    print('\n\n--------------------\n\n')
                    print(source_sentences)
                    print('\n-----------------------\n')
                continue
        target_indices.append(torch.Tensor([len(source_sentences)]))                        # Appended the EOS token
        
        return torch.stack(target_indices), self.source_sentences_path[idx], self.target_sentences_path[idx], len(target_indices)

    def get_index(self, source_sentences, target_sentence):
        for idx, sent in enumerate(source_sentences):
            if target_sentence in sent:
                return idx

    def is_blank_sentence(self, sentence):
        for token in sentence:
            if self.lemmatizer.lemmatize(token) in self.words_set:
                return False
        return True

def collator(DataLoaderBatch):
    """
    Function to generate batch-wise padding of the data (dynamic padding).
    Args: list of tuple(src_seqs, vid_feature_seqs, aud_feature_seqs, tgt_seqs)
        - src_seq: torch tensor of variable length
        - vid_feat_seqs: 2D torch tensor of variable frame len in videos
        - aud_feat_seqs: 2D torch tensor of variable frame len in audios
        - tgt_seq: torch tensor of variable length
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length)
        src_lengths: list of length (batch_size) which is the original length for each padded src seq
        vid_feat_seqs: torch tensor of shape (batch_size, padded_vid_seq_len, img_feat_size)
        vid_lengths: list of length (batch_size) which is the original length for each padded video
        aud_feat_seqs: torch tensor of shape (batch_size, padded_aud_seq_len, aud_feat_size)
        aud_lengths: list of length (batch_size) which is the original length for each padded audio
        tgt_seqs: torch tensor of shape (batch_size, padded_length)
        tgt_lengths: list of length (batch_size) which is the original length for each padded tgt seq
        
    code adapted from : https://github.com/yunjey/seq2seq-dataloader/
    """
    def pad_get_len(items):
        lengths = [num_elements.size(0) for num_elements in items]
        padded_seq = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)
        return padded_seq, lengths

    # separate the values returned from the dataset
    src_seqs, vid_feat_seqs, aud_feat_seqs, tgt_seqs = zip(*DataLoaderBatch)

    # pad the tensors and create batches
    src_seqs, src_lengths = pad_get_len(src_seqs)
    vid_feat_seqs, vid_lengths = pad_get_len(vid_feat_seqs)
    aud_feat_seqs, aud_lengths = pad_get_len(aud_feat_seqs)
    tgt_seqs, tgt_lengths = pad_get_len(tgt_seqs)

    return src_seqs, src_lengths, vid_feat_seqs, vid_lengths, aud_feat_seqs, aud_lengths, tgt_seqs, tgt_lengths

def target_collator(DataLoaderBatch):
    batch_items = [item for item in DataLoaderBatch]
    items, source_sent_paths, target_sent_paths, _ = zip(*batch_items)
    lengths = [len(num_target_sent) for num_target_sent in items]
    padded_seq = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)
    return padded_seq, source_sent_paths, target_sent_paths, lengths

def gen_train_val_indices(dataset, validation_split=0.1, shuffle=True):
    # Ignore indices from test set and videos where ground-truth is missing
    test_indices = get_test_indices()
    with open('none_idxs.pkl', 'rb') as f:
        none_indices = pickle.load(f)

    dataset_size = len(dataset)
    indices = [idx for idx in range(dataset_size) if idx not in test_indices and idx not in none_indices]
    split = int(np.floor(validation_split * len(indices)))

    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices

def get_test_indices():
    with open('test_indices.pkl', 'rb') as f:
        test_indices = pickle.load(f)
    return test_indices

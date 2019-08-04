"""
Train a model on the MMS Dataset.
"""
import copy
import logging
import os
import pickle
import random
from collections import OrderedDict
from json import dumps

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from datasets import *
from models import MMBiDAF
from PIL import Image
from rouge import Rouge
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from nltk.tokenize import sent_tokenize


def main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, batch_size=3, num_epochs=100):
    # Get sentence embeddings
    train_text_loader = torch.utils.data.DataLoader(TextDataset(course_dir, max_text_length), batch_size = batch_size, shuffle = False, num_workers = 2, collate_fn=collator)

    # Get Audio embeddings
    train_audio_loader = torch.utils.data.DataLoader(AudioDataset(course_dir), batch_size = batch_size, shuffle = False, num_workers = 2, collate_fn=collator)

    # Preprocess the image in prescribed format
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    train_image_loader = torch.utils.data.DataLoader(ImageDataset(course_dir, transform), batch_size = batch_size, shuffle = False, num_workers = 2, collate_fn=collator)

    # Load Target text
    train_target_loader = torch.utils.data.DataLoader(TargetDataset(course_dir), batch_size = batch_size, shuffle = False, num_workers = 2, collate_fn=target_collator)

    # Create model
    model = MMBiDAF(hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), 1e-4)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Let's do this!
    step = 0
    model.train()
    hidden_state = None
    epoch = 0
    loss = 0
    eps = 1e-8

    with torch.enable_grad(), tqdm(total=max(len(train_text_loader.dataset), len(train_image_loader.dataset), len(train_audio_loader.dataset))) as progress_bar:
        for (batch_text, original_text_lengths), (batch_audio, original_audio_lengths), (batch_images, original_img_lengths), (batch_target_indices, source_path, target_path, original_target_len) in zip(train_text_loader, train_audio_loader, train_image_loader, train_target_loader):
            loss = 0
            # Setup for forward
            batch_size = batch_text.size(0)
            optimizer.zero_grad()
            epoch += 1

            # Forward
            out_distributions, loss = model(batch_text, original_text_lengths, batch_audio, original_audio_lengths, batch_images, original_img_lengths, batch_target_indices, original_target_len)
            loss_val = loss.item()           # numerical value of loss

            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Generate summary
            print('Generated summary for iteration {}: '.format(epoch))
            summary = get_generated_summary(out_distributions, original_text_length, source_path)
            print(summary)
            
            # Evaluation
            rouge = Rouge()
            rouge_scores = rouge.get_scores(source_path, target_path, avg=True)
            print('Rouge score at iteration {} is {}: '.format(epoch, rouge_scores))

            # Generate Output Heatmaps
            sns.set()
            for idx in range(len(out_distributions)):
                out_distributions[idx] = out_distributions[idx].squeeze(0).detach().numpy()      # Converting each timestep distribution to numpy array
            out_distributions = np.asarray(out_distributions)   # Converting the timestep list to array
            ax = sns.heatmap(out_distributions)
            fig = ax.get_figure()
            fig.savefig(out_heatmaps_dir + str(epoch) + '.png')


def get_generated_summary(out_distributions, original_text_length, source_path):
    out_distributions = np.array([dist[0].cpu().detach().numpy() for dist in out_distributions])  # TODO: Batch 0
    generated_summary = []
    for timestep, probs in enumerate(out_distributions):
        if(probs[int(original_text_length)] == np.argmax(probs)):
            break
        else:
            max_prob_idx = np.argmax(probs, 0)
            generated_summary.append(get_source_sentence(source_path[0], max_prob_idx-1))

            # Setting the generated sentence's prob to zero in the remaining timesteps - coverage?
            out_distributions[:, max_prob_idx] = 0
    
    return generated_summary

def get_source_sentence(source_path, idx):
    lines = []
    try:
        with open(source_path) as f:
            for line in f:
                    if re.match(r'\d+:\d+', line) is None:
                        line = line.replace('[MUSIC]', '')
                        lines.append(line.strip())
    except Exception as e:
        logging.error('Unable to open file. Exception: ' + str(e))
    else:
        source_text = ' '.join(lines)
        source_sentences = sent_tokenize(source_text)
        for i in range(len(source_sentences)):
            source_sentences[i] = source_sentences[i].lower()
        return source_sentences[idx]

def collator(DataLoaderBatch):
    items = [item[0] for item in DataLoaderBatch]
    lengths = [num_elements.size(0) for num_elements in items]
    padded_seq = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)
    return padded_seq, lengths

def target_collator(DataLoaderBatch):
    batch_items = [item for item in DataLoaderBatch]
    items, source_sent_paths, target_sent_paths, _ = zip(*batch_items)
    lengths = [len(num_target_sent) for num_target_sent in items]
    padded_seq = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)
    return padded_seq, source_sent_paths, target_sent_paths, lengths

if __name__ == '__main__':
    course_dir = '/home/anish17281/NLP_Dataset/dataset/'
    text_embedding_size = 300
    audio_embedding_size = 128
    image_embedding_size = 1000
    hidden_size = 100
    drop_prob = 0.2
    max_text_length = 405
    num_epochs = 100
    batch_size = 3
    out_heatmaps_dir = '/home/amankhullar/model/output_heatmaps/'
    main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, batch_size, num_epochs)

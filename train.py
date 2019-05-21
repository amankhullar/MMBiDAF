"""
Train a model on the MMS Dataset.
"""
import logging
import os
import pickle
import random
from collections import OrderedDict
from json import dumps

import numpy as np
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
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load


def main(course_dir, text_embedding_size, audio_embedding_size, hidden_size, drop_prob, max_text_length, num_epochs=100):
    # Get sentence embeddings
    train_text_loader = torch.utils.data.DataLoader(TextDataset(course_dir, max_text_length), batch_size = 1, shuffle = False, num_workers = 2)

    # Get Audio embeddings
    train_audio_loader = torch.utils.data.DataLoader(AudioDataset(course_dir), batch_size = 1, shuffle = False, num_workers = 2)
    
    # Preprocess the image in prescribed format
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    train_image_loader = torch.utils.data.DataLoader(ImageDataset(course_dir, transform), batch_size = 1, shuffle = False, num_workers = 2)

    # Load Target text
    train_target_loader = torch.utils.data.DataLoader(TargetDataset(course_dir), batch_size = 1, shuffle = False, num_workers = 2)

    # Create model
    model = MMBiDAF(hidden_size, text_embedding_size, audio_embedding_size, drop_prob, max_text_length)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), 1e-4)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Let's do this!
    step = 0
    model.train()
    model.float()
    hidden_state = None
    epoch = 0
    loss = 0
    eps = 1e-8

    with torch.enable_grad(), tqdm(total=max(len(train_text_loader.dataset), len(train_image_loader.dataset), len(train_audio_loader.dataset))) as progress_bar:
        for (batch_text, original_text_length), batch_audio, batch_images, batch_target_indices in zip(train_text_loader, train_audio_loader, train_image_loader, train_target_loader):
            loss = 0
            # Setup for forward
            batch_size = batch_text.size(0)
            optimizer.zero_grad()
            epoch += 1
            # Required for debugging
            batch_text = batch_text.float()
            batch_audio = batch_audio.float()
            batch_images = batch_images.float()

            # Forward
            out_distributions = model(batch_text, original_text_length, batch_audio, torch.Tensor([batch_audio.size(1)]), batch_images, torch.Tensor([batch_images.size(1)]), hidden_state)
            
            for batch, target_indices in enumerate(batch_target_indices):
                for timestep, target_idx in enumerate(target_indices.squeeze(1)):
#                     print(target_idx)
                    prob = out_distributions[timestep][batch, int(target_idx)]
#                     print("Prob = {}".format(prob))
                    loss += -1 * torch.log(prob + eps)
#                     print("Loss = {}".format(loss))

            # Backward
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            print('Loss for Epoch {} : '.format(epoch))
            #     print(out_distributions)
#             print(len(out_distributions))
#             print(out_distributions[0].size())
            print(loss)
#             break


    
if __name__ == '__main__':
    course_dir = '/home/anish17281/NLP_Dataset/dataset/'
    text_embedding_size = 300
    audio_embedding_size = 128
    hidden_size = 100
    drop_prob = 0.2
    max_text_length = 405
    num_epochs = 100
    main(course_dir, text_embedding_size, audio_embedding_size, hidden_size, drop_prob, max_text_length, num_epochs)

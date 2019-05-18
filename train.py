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
from datasets import AudioDataset, ImageDataset, TextDataset
from models import MMBiDAF
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load


def main(course_dir, text_embedding_size, audio_embedding_size, hidden_size, drop_prob, max_text_length):
    # Get sentence embeddings
    train_text_loader = torch.utils.data.DataLoader(TextDataset(course_dir, max_text_length), batch_size = 1, shuffle = False, num_workers = 2)

    # Get Audio embeddings
    train_audio_loader = torch.utils.data.DataLoader(AudioDataset(course_dir), batch_size = 1, shuffle = False, num_workers = 2)
    
    # Preprocess the image in prescribed format
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    train_image_loader = torch.utils.data.DataLoader(ImageDataset(course_dir, transform), batch_size = 1, shuffle = False, num_workers = 2)

    # Create model
    model = MMBiDAF(hidden_size, text_embedding_size, audio_embedding_size, drop_prob, max_text_length)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), 1e-3)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Let's do this!
    step = 0
    model.train()
    model.float()
    epoch = 0

    with torch.enable_grad(), tqdm(total=max(len(train_text_loader.dataset), len(train_image_loader.dataset), len(train_audio_loader.dataset))) as progress_bar:
        for (batch_text, original_text_length), batch_audio, batch_images in zip(train_text_loader, train_audio_loader, train_image_loader):
                optimizer.zero_grad()
                
                batch_text = batch_text.float()
                batch_audio = batch_audio.float()
                batch_images = batch_images.float()
                
                hidden_state, final_out, sentence_dist = model(batch_text, torch.Tensor([batch_text.size(1)]), batch_audio, torch.Tensor([batch_audio.size(1)]), batch_images, torch.Tensor([batch_images.size(1)]))
                
                print(final_out)
                
                break
                # Forward
                # log_p1, log_p2 = model(cw_idxs, qw_idxs)
                # y1, y2 = y1.to(device), y2.to(device)
                # loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                # loss_val = loss.item()

                # # Backward
                # loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer.step()
                # scheduler.step(step // batch_size)
                # ema(model, step // batch_size)


    
if __name__ == '__main__':
    course_dir = '/home/anish17281/NLP_Dataset/dataset/'
    text_embedding_size = 300
    audio_embedding_size = 128
    hidden_size = 100
    drop_prob = 0.2
    max_text_length = 405
    main(course_dir, text_embedding_size, audio_embedding_size, hidden_size, drop_prob, max_text_length)

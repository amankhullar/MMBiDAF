import os
import sys
import numpy as np

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
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from nltk.tokenize import sent_tokenize
from json import dumps

import util
from args import get_train_args

def main(hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    model = MMBiDAF(hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, device, drop_prob, max_text_length)
    model = nn.DataParallel(model, gpu_ids)
    
    
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, checkpoint_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()
    # the loading is being performed in the train.py file as well

    # model = load_model(model, checkpoint_path, args.gpu_ids)
    print("Model Loaded")
    print(model)

if __name__ == "__main__":
    hidden_size = 100
    text_embedding_size = 300
    audio_embedding_size = 128
    image_embedding_size = 1000
    drop_prob = 0.2
    max_text_length = 405
    args = get_train_args()
    checkpoint_path = "/home/amankhullar/model/multimodal_bidaf/save/train/temp-01/step_22800.pth.tar"
    main(hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path)


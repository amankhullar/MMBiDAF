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

def get_test_indices():
    with open('test_indices.pkl', 'rb') as f:
        test_indices = pickle.load(f)
    return test_indices

def evaluate(courses_dir, hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path, batch_size=3):
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

    # Create Dataset objects
    text_dataset = TextDataset(courses_dir, max_text_length)
    audio_dataset = AudioDataset(courses_dir)
    target_dataset = TargetDataset(courses_dir)
    # Preprocess the image in prescribed format
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    image_dataset = ImageDataset(courses_dir, transform)

    # Creating data indices for training and validation splits:
    test_indices = get_test_indices()

    # Creating PT data sampler and loaders:
    test_sampler = torch.utils.data.SequentialSampler(test_indices)

    # Get sentence embeddings
    test_text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=test_sampler)

    # Get Audio embeddings
    test_audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=test_sampler)

    # Get images
    test_image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=test_sampler)

    # Load Target text
    test_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=target_collator, sampler=test_sampler)

    batch_idx = 0
    for (batch_text, original_text_lengths), (batch_audio, original_audio_lengths), (batch_images, original_img_lengths), (batch_target_indices, batch_source_paths, batch_target_paths, original_target_len) in zip(test_text_loader, test_audio_loader, test_image_loader, test_target_loader):
        batch_idx += 1

        max_dec_len = torch.max(original_target_len)             # TODO check error : max decoder timesteps for each batch 

        # Transfer tensors to GPU
        batch_text = batch_text.to(device)
        log.info("Loaded batch text")
        batch_audio = batch_audio.to(device)
        log.info("Loaded batch audio")
        batch_images = batch_images.to(device)
        log.info("Loaded batch image")
        batch_target_indices = batch_target_indices.to(device)
        log.info("Loaded batch targets")

        batch_out_distributions, _ = model(batch_text, original_text_lengths, batch_audio, original_audio_lengths, batch_images, original_img_lengths, batch_target_indices, original_target_len, max_dec_len)

        # Generate summary for current batch
        print('Generated summary for batch {}: '.format(batch_idx))
        summaries = get_generated_summaries(batch_out_distributions, original_text_lengths, batch_source_paths)
        print(summaries)

def get_generated_summaries(batch_out_distributions, original_text_lengths, batch_source_paths, method='beam', k=5):
    batch_out_distributions = np.array([dist.cpu().detach().numpy() for dist in batch_out_distributions])
    generated_summaries = []

    for batch_idx in range(batch_out_distributions.shape[0]):
        out_distributions = batch_out_distributions[batch_idx, :, :]

        if method == 'beam':
            summaries = beam_search(out_distributions, original_text_lengths[batch_idx], batch_source_paths[batch_idx], k=k)
        else:
            generated_summary = greedy_search(out_distributions, original_text_lengths[batch_idx], batch_source_paths[batch_idx])
            summaries = [generated_summary]
        generated_summaries.append(summaries)

    return generated_summaries

def greedy_search(out_distributions, original_text_length, source_path):
    generated_summary = []
    for probs in out_distributions: # Looping over timesteps
        if(probs[int(original_text_length)] == np.argmax(probs)): # EOS
            break
        max_prob_idx = np.argmax(probs, 0)
        generated_summary.append(get_source_sentence(source_path, max_prob_idx))
        # Setting the generated sentence's prob to zero in the remaining timesteps - coverage?
        # out_distributions[:, max_prob_idx] = 0
    return generated_summary

def beam_search(out_distributions, original_text_length, source_path, k=5):
    eps = 1e-8
    sequences = [[list(), 1.0]]
    # Loop over timesteps
    for probs in out_distributions:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(probs)):
                candidate = [seq + [j], score * -np.log(probs[j]+eps)]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    
    beam_summaries = []
    for seq in sequences:
        generated_summary = []
        for sent_idx in seq[0]:
            if sent_idx == original_text_length:
                break
            generated_summary.append(get_source_sentence(source_path, sent_idx))
        beam_summaries.append(generated_summary)

    return beam_summaries

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

if __name__ == "__main__":
    hidden_size = 100
    text_embedding_size = 300
    audio_embedding_size = 128
    image_embedding_size = 1000
    drop_prob = 0.2
    max_text_length = 405
    args = get_train_args()
    checkpoint_path = "/home/amankhullar/model/multimodal_bidaf/save/train/temp-01/step_22800.pth.tar"
    courses_dir = '/home/anish17281/NLP_Dataset/dataset/'
    evaluate(courses_dir, hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path)


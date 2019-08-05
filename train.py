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
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from nltk.tokenize import sent_tokenize

import util
from args import get_train_args


def gen_train_val_indices(dataset, validation_split=0.1, shuffle=True):
    with open('test_indices.pkl', 'rb') as f:
        test_indices = pickle.load(f)
    dataset_size = len(dataset)
    print("dataset_size: " + str(dataset_size))
    indices = [idx for idx in range(dataset_size) if idx not in test_indices]
    print("indices len: " + str(len(indices)))
    split = int(np.floor(validation_split * len(indices)))
    if shuffle :
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices

def main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, args, batch_size=3, num_epochs=100):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create Dataset objects
    text_dataset = TextDataset(course_dir, max_text_length)
    audio_dataset = AudioDataset(course_dir)
    target_dataset = TargetDataset(course_dir)
    # Preprocess the image in prescribed format
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    image_dataset = ImageDataset(course_dir, transform)

    assert len(text_dataset) == len(audio_dataset) and len(audio_dataset) == len(image_dataset) and len(image_dataset) == len(target_dataset), "Unequal dataset lengths"

    # Creating data indices for training and validation splits:
    train_indices, val_indices = gen_train_val_indices(text_dataset)

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SequentialSampler(train_indices)
    val_sampler = torch.utils.data.SequentialSampler(val_indices)

    # Get sentence embeddings
    train_text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    val_text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # Get Audio embeddings
    train_audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    val_audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # Get images
    train_image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    val_image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # Load Target text
    train_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=target_collator, sampler=train_sampler)
    val_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=target_collator, sampler=val_sampler)

    # print("lens - train_text_loader {}, val_text_loader {}".format(len(train_text_loader), len(val_text_loader)))
    # print("lens - train_audio_loader {}, val_audio_loader {}".format(len(train_audio_loader), len(val_audio_loader)))
    # print("lens - train_image_loader {}, val_image_loader {}".format(len(train_image_loader), len(val_image_loader)))
    # print("lens - train_target_loader {}, val_target_loader {}".format(len(train_target_loader), len(val_target_loader)))

    # Create model
    model = MMBiDAF(hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, device, drop_prob, max_text_length)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)           # For exponential moving average

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)           # Need to change the metric name

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr, weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # TODO : Add the dev and the test dataset loaders

    # Let's do this!
    loss = 0
    eps = 1e-8
    log.info("Training...")
    steps_till_eval = args.eval_steps
    epoch = step // len(TextDataset(course_dir, max_text_length))

    while epoch != args.num_epochs:
        epoch += 1
        log.info("Starting epoch {epoch}...")
        with torch.enable_grad(), tqdm(total=len(train_text_loader.dataset)) as progress_bar:
            for (batch_text, original_text_lengths), (batch_audio, original_audio_lengths), (batch_images, original_img_lengths), (batch_target_indices, source_path, target_path, original_target_len) in zip(train_text_loader, train_audio_loader, train_image_loader, train_target_loader):
                loss = 0

                # Transfer tensors to GPU
                batch_text = batch_text.to(device)
                log.info("Loaded batch text")
                batch_audio = batch_audio.to(device)
                log.info("Loaded batch audio")
                batch_images = batch_images.to(device)
                log.info("Loaded batch image")
                batch_target_indices = batch_target_indices.to(device)
                log.info("Loaded batch targets")

                # Setup for forward
                batch_size = batch_text.size(0)
                optimizer.zero_grad()
                
                log.info("Starting forward pass")
                # Forward
                out_distributions, loss = model(batch_text, original_text_lengths, batch_audio, original_audio_lengths, batch_images, original_img_lengths, batch_target_indices, original_target_len)
                loss_val = loss.item()           # numerical value of loss
                
                log.info("Starting backward")

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)        # To tackle exploding gradients
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                
                print("Reached here")
                sys.exit()

                # Generate summary
                print('Generated summary for iteration {}: '.format(epoch))
                summary = get_generated_summary(out_distributions, original_text_lengths, source_path)
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
    args = get_train_args()
    main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, args, batch_size, num_epochs)

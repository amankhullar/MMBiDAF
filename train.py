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

import evaluate
import util
from args import get_train_args

def main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, args, checkpoint_path=None, batch_size=3, num_epochs=100, USE_CPU=False):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    if USE_CPU:
        device = torch.device('cpu')
        args.gpu_ids = []
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create Train Dataset objects
    train_text_path = os.path.join(course_dir, 'text', 'sum_train', 'tran.tok.txt')
    gt_path = os.path.join(course_dir, 'text', 'sum_train', 'desc.tok.txt')
    word_idx_path = os.path.join(course_dir, 'temp', 'word_idx.pkl')
    vid_feat_path = os.path.join(course_dir, 'video_action_features')
    aud_feat_path = os.path.join(course_dir, 'audio_feat')      # TODO : check path
    multimodal_dataset = MultimodalDataset(train_text_path, gt_path, word_idx_path, vid_feat_path, vid_feat_path, max_text_length)        # TODO : add the audio features instead of video features

    # audio_dataset = AudioDataset(course_dir)
    # target_dataset = TargetDataset(course_dir)
    # # Preprocess the image in prescribed format
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    # image_dataset = ImageDataset(course_dir, transform)

    # assert len(text_dataset) == len(audio_dataset) and len(audio_dataset) == len(image_dataset) and len(image_dataset) == len(target_dataset), "Unequal dataset lengths"

    # Creating data indices for training and validation splits:
    # train_indices, val_indices = gen_train_val_indices(text_dataset)

    # # Creating PT data samplers and loaders:
    # train_sampler = torch.utils.data.SequentialSampler(train_indices)
    # val_sampler = torch.utils.data.SequentialSampler(val_indices)


    # Load Data Loader
    train_loader = torch.utils.data.DataLoader(multimodal_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collator)

    # Get sentence embeddings
    # train_text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    # val_text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # # Get Audio embeddings
    # train_audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    # val_audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # # Get images
    # train_image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=train_sampler)
    # val_image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator, sampler=val_sampler)

    # # Load Target text
    # train_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=target_collator, sampler=train_sampler)
    # val_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=target_collator, sampler=val_sampler)

    # print("lens - train_text_loader {}, val_text_loader {}".format(len(train_text_loader), len(val_text_loader)))
    # print("lens - train_audio_loader {}, val_audio_loader {}".format(len(train_audio_loader), len(val_audio_loader)))
    # print("lens - train_image_loader {}, val_image_loader {}".format(len(train_image_loader), len(val_image_loader)))
    # print("lens - train_target_loader {}, val_target_loader {}".format(len(train_target_loader), len(val_target_loader)))

    # Create model
    audio_embedding_size = image_embedding_size # TODO : Remove this statement after getting audio features
    word_vectors = torch.load(os.path.join(course_dir, 'temp', 'word_vectors.pt'))
    if checkpoint_path == None:
        model = MMBiDAF(hidden_size, word_vectors, text_embedding_size, audio_embedding_size, image_embedding_size, device, drop_prob, max_text_length)
    else:
        model = util.load_model(model, checkpoint_path, device, args.gpu_ids, return_step=False)
    #if not USE_CPU:
    #    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, device, args.gpu_ids)
    else:
        step = 0
    if not USE_CPU:
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

    # Let's do this!
    loss = 0
    eps = 1e-8
    log.info("Training...")
    steps_till_eval = args.eval_steps
    epoch = step // len(multimodal_dataset)

    while epoch != args.num_epochs:
        epoch += 1
        log.info("Starting epoch {epoch}...")
        count_item = 0
        loss_epoch = 0
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for batch_text, original_text_lengths, batch_images, original_img_lengths, batch_audio, original_audio_lengths, batch_target_indices, original_target_len in train_loader:
                count_item += 1
                loss = 0
                max_dec_len = max(original_target_len)        
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
                batch_out_distributions, loss = model(batch_text, original_text_lengths, batch_audio, original_audio_lengths, batch_images, original_img_lengths, batch_target_indices, original_target_len, max_dec_len)
                loss_val = loss.item()           # numerical value of loss
                loss_epoch = loss_epoch + loss_val
                
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
                
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    model.eval()
                    
                    # Evaluate the model on the training set
                    evaluate.evaluate(course_dir, hidden_size, text_embedding_size, audio_embedding_size,\
                        image_embedding_size, drop_prob, max_text_length, args, checkpoint_path='', batch_size=1,\
                        model=model)

                    model.train()
                    saver.save(step, model, device)
                    ema.resume(model)

                # Generate summary
                # print('Generated summary for iteration {}: '.format(epoch))
                # summaries = get_generated_summaries(batch_out_distributions, original_text_lengths, batch_source_paths)
                # print(summaries)
                
                # Evaluation
                # rouge = Rouge()
                # rouge_scores = rouge.get_scores(batch_source_paths, batch_target_paths, avg=True)
                # print('Rouge score at iteration {} is {}: '.format(epoch, rouge_scores))

                # Generate Output Heatmaps
                # sns.set()
                # for idx in range(len(out_distributions)):
                #     out_distributions[idx] = out_distributions[idx].squeeze(0).detach().numpy()      # Converting each timestep distribution to numpy array
                # out_distributions = np.asarray(out_distributions)   # Converting the timestep list to array
                # ax = sns.heatmap(out_distributions)
                # fig = ax.get_figure()
                # fig.savefig(out_heatmaps_dir + str(epoch) + '.png')
            print("Epoch loss is : {}".format(loss_epoch/count_item))

if __name__ == '__main__':
    course_dir = '/home/aman_khullar/how2/'
    checkpoint_path = "/home/aman_khullar/multimodal/MMBiDAF/save/train/temp-05/step_90017.pth.tar"
    text_embedding_size = 100
    audio_embedding_size = 43
    image_embedding_size = 2048
    hidden_size = 100
    drop_prob = 0.2
    max_text_length = 210100   # calculated using a simple script to go over all the transcripts and count tokens in list split by (' ') and ignored filename
    num_epochs = 90
    batch_size = 3
    out_heatmaps_dir = '/home/amankhullar/model/output_heatmaps/'
    USE_CPU = False          # To check if the error being encountered is that of CUDA
    args = get_train_args()
    main(course_dir, text_embedding_size, audio_embedding_size, image_embedding_size, hidden_size, drop_prob, max_text_length, out_heatmaps_dir, args, checkpoint_path, batch_size, num_epochs, USE_CPU)

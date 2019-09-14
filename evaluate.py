import os
import pickle
import re
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

from rouge import Rouge
from nltk import PorterStemmer

from sklearn.metrics import f1_score

stemmer = PorterStemmer()

USE_CPU = False

def get_indices(dataset):
    # return get_test_indices()
    train_indices, _ = gen_train_val_indices(dataset)
    return train_indices

def evaluate(course_dir, hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path, batch_size=1,\
             model=None, test_text_loader=None, test_audio_loader=None, test_image_loader=None, test_target_loader=None):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    
    device, args.gpu_ids = util.get_available_devices()
    device = torch.device(f'cuda:{args.gpu_ids[-1]}')
    print("GPU device is : {}".format(device))

    if USE_CPU:
        device = torch.device('cpu')    #### TODO : only because GPU is out of memory
        args.gpu_ids = []    #### TODO : Gpu out of memory
    
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create Test Dataset objects
    train_text_path = os.path.join(course_dir, 'text', 'sum_devtest', 'tran.tok.txt')
    gt_path = os.path.join(course_dir, 'text', 'sum_devtest', 'desc.tok.txt')
    word_idx_path = os.path.join(course_dir, 'temp', 'word_idx.pkl')
    idx_word_path = os.path.join(course_dir, 'temp', 'idx_word.pkl')
    vid_feat_path = os.path.join(course_dir, 'video_action_features')
    aud_feat_path = os.path.join(course_dir, 'audio_feat')      # TODO : check path
    multimodal_dataset = MultimodalDataset(train_text_path, gt_path, word_idx_path, vid_feat_path, vid_feat_path, max_text_length)        # TODO : add the audio features instead of video features

    if model == None:
        audio_embedding_size = image_embedding_size # TODO : Remove this statement after getting audio features
        word_vectors = torch.load(os.path.join(course_dir, 'temp', 'word_vectors.pt'))
        model = MMBiDAF(hidden_size, word_vectors, text_embedding_size, audio_embedding_size, image_embedding_size, device, drop_prob, max_text_length)
    #model = nn.DataParallel(model, gpu_ids)
    
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, checkpoint_path, device, args.gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
        # the loading is being performed in the train.py file as well

        # model = load_model(model, checkpoint_path, args.gpu_ids)
        print("Model Loaded")
#     print(model)

    # Load Data Loader
    test_loader = torch.utils.data.DataLoader(multimodal_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collator)

    batch_idx = 0
    total_scores = [0]*9        # in order of 'p' 'r' and 'f' for r1, r2, rl
    f1_score = 0
    with torch.no_grad(), tqdm(total=len(test_loader.dataset)) as progress_bar:
        for batch_text, original_text_lengths, batch_images, original_img_lengths, batch_audio, original_audio_lengths, batch_target_indices, original_target_len in test_loader:
            batch_idx += 1

            max_dec_len = max(original_target_len)             # TODO check error : max decoder timesteps for each batch 

            # Transfer tensors to GPU
            batch_text = batch_text.to(device)
            # log.info("Loaded batch text")
            batch_audio = batch_audio.to(device)
            # log.info("Loaded batch audio")
            batch_images = batch_images.to(device)
            # log.info("Loaded batch image")
            batch_target_indices = batch_target_indices.to(device)
            # log.info("Loaded batch targets")

            batch_out_distributions, _ = model(batch_text, original_text_lengths, batch_audio, original_audio_lengths, \
                                            batch_images, original_img_lengths, batch_target_indices, original_target_len, max_dec_len)

            # Generate summary for current batch
            log.info("\n\nGenerating summaries for batch {}\n".format(batch_idx))

            summaries, gen_idxs = get_generated_summaries(batch_out_distributions, original_text_lengths, idx_word_path) # (batch_size, beam_size, sents)

            print('Generated summaries for batch {}: '.format(batch_idx))
            print(summaries)

            try:
            # Calculate Rouge score for the current batch
                all_scrores = compute_rouge(summaries, batch_target_indices, idx_word_path, beam_size=1) # tuple of all scores
                for idx, score in enumerate(all_scrores):
                    total_scores[idx] += score

                # Calculate F1 score for the current batch
                curr_f1_score = compute_f1(gen_idxs, batch_target_indices, beam_size=1)
                print("F1 score: {}".format(curr_f1_score))
                f1_score += curr_f1_score
            except Exception as e:
                print("Error: " + str(e))
                continue

        # Average Rouge score across batches
        for idx in range(len(total_scores)):                
            total_scores[idx] /= batch_idx

        #Average F1 score
        f1_score /= batch_idx

        print("Average Rouge score on the data is : {}".format(total_scores))
        print("Average F1 score on the data is : {}".format(f1_score))

def get_generated_summaries(batch_out_distributions, original_text_lengths, idx_word_path, method='greedy', k=5):
    batch_out_distributions = np.array([dist.cpu().detach().numpy() for dist in batch_out_distributions])
    generated_summaries = []
    gen_idxs = []
    for batch_idx in range(batch_out_distributions.shape[0]):
        out_distributions = batch_out_distributions[batch_idx, :, :]

        if method == 'beam':
            summaries, idxs = beam_search(out_distributions, original_text_lengths[batch_idx], idx_word_path, k=k)
        else:
            generated_summary, idxs = greedy_search(out_distributions, original_text_lengths[batch_idx], idx_word_path)
            summaries = [generated_summary]
            idxs = [idxs]
        generated_summaries.append(summaries)
        gen_idxs.append(idxs)

    return generated_summaries, gen_idxs

def greedy_search(out_distributions, original_text_length, idx_word_path):
    generated_summary = []
    gen_idxs = []
    for probs in out_distributions: # Looping over timesteps
        word_idx = np.argmax(probs, 0)
        try:
            with open(idx_word_path, 'rb') as f:
                word_idx_dict = pickle.load(f)
                word = word_idx_dict[word_idx]
        except Exception as e:
            print("Could not load idx to word file with exception as : " + str(e))
            sys.exit()
        if word == '<END>':
            break
        if word != '<START>':
            generated_summary.append(word)
        gen_idxs.append(word_idx)
        # Setting the generated sentence's prob to zero in the remaining timesteps - coverage?
        # out_distributions[:, max_prob_idx] = 0
    generated_summary = ' '.join(generated_summary)
    return generated_summary, gen_idxs

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
    beam_idxs = []
    for seq in sequences:
        generated_summary = []
        gen_idxs = []
        for sent_idx in seq[0]:
            if sent_idx == original_text_length:
                break
            generated_summary.append(get_source_sentence(source_path, sent_idx))
            gen_idxs.append(sent_idx)
        beam_summaries.append(generated_summary)
        beam_idxs.append(gen_idxs)

    return beam_summaries, beam_idxs

def get_source_sentence(source_path, idx):
    lines = []
    try:
        with open(source_path, 'rb') as f:
            source_file = pickle.load(f)
    except Exception as e:
        logging.error('Unable to open file. Exception: ' + str(e))
    else:
        source_sentences = [sent[0] for sent in source_file]
#         for i in range(len(source_sentences)):
#             source_sentences[i] = source_sentences[i].lower()
        if idx == len(source_sentences):
            return 0
        if idx > len(source_sentences):
            return None
        return source_sentences[idx]

def prepare(gt, res):
    clean_gt = [" ".join([stemmer.stem(i) for i in line.split()]) for line in gt]
    clean_res = [" ".join([stemmer.stem(i) for i in line.split()]) for line in res]
    return gt, res

def get_rouge(clean_gt, clean_res):
    rouge = Rouge()
    scores = rouge.get_scores(clean_res, clean_gt, avg=True)
    return scores

def get_target_sum(batch_target_indices, idx_word_path):
    try:
        with open(idx_word_path, 'rb') as f:
            idx_word_dict = pickle.load(f)
    except Exception as e:
        print("Could not load idx_word dictionary with error : " + e)
        sys.exit()
    gt_smry = []
    for word_idx in batch_target_indices:
        word = idx_word_dict[int(word_idx)]
        if word != '<OOV>' and word != '<START>' and word != '<END>':
            gt_smry.append(idx_word_dict[int(word_idx)])
    gt_smry = ' '.join(gt_smry)
    return gt_smry

def compute_rouge(summaries, batch_target_indices, idx_word_path, beam_size=1):
    total_score_r1p = total_score_r1r = total_score_r1f = total_score_r2p = \
    total_score_r2r = total_score_r2f = total_score_rlp = total_score_rlr = total_score_rlf = 0
    batch_count = len(summaries)
    print("Summaries is : {}".format(summaries))
    for batch_idx, batch_val in enumerate(summaries):
        gt_summary = get_target_sum(batch_target_indices[batch_idx], idx_word_path)
        res_data = batch_val[beam_size-1]
        clean_gt, clean_res = prepare(gt_summary, res_data)
        score = get_rouge(clean_gt, clean_res)
        total_score_r1p += score['rouge-1']['p']
        total_score_r1r += score['rouge-1']['r']
        total_score_r1f += score['rouge-1']['f']
        total_score_r2p += score['rouge-2']['p']
        total_score_r2r += score['rouge-2']['r']
        total_score_r2f += score['rouge-2']['f']
        total_score_rlp += score['rouge-l']['p']
        total_score_rlr += score['rouge-l']['r']
        total_score_rlf += score['rouge-l']['f']
    return (total_score_r1p/batch_count, total_score_r1r/batch_count, total_score_r1f/batch_count, \
        total_score_r2p/batch_count, total_score_r2r/batch_count, total_score_r2f/batch_count, \
        total_score_rlp/batch_count, total_score_rlr/batch_count, total_score_rlf/batch_count)

def compute_f1(gen_idxs, batch_target_indices, beam_size=1):
    f1_val = 0
    batch_count = len(gen_idxs)
    for batch_idx, batch_val in enumerate(gen_idxs):
        gt_idxs = batch_target_indices[batch_idx].squeeze(0).tolist()
        res_idxs = batch_val[beam_size-1]
        min_len = min(len(gt_idxs), len(res_idxs))
        gt_idxs = gt_idxs[:min_len]
        res_idxs = res_idxs[:min_len]
        f1_val += f1_score(gt_idxs, res_idxs, average='macro')          # this average is used in SQUAD dataset
    return f1_val/batch_count

if __name__ == "__main__":
    hidden_size = 100
    text_embedding_size = 100
    audio_embedding_size = 43
    image_embedding_size = 2048
    drop_prob = 0.2
    max_text_length = 210100
    args = get_train_args()
    checkpoint_path = "/home/aman_khullar/multimodal/MMBiDAF/save/train/temp-05/step_90017.pth.tar"
    courses_dir = '/home/aman_khullar/how2/'
    evaluate(courses_dir, hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob, max_text_length, args, checkpoint_path)


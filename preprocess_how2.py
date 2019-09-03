import os
import pickle
import sys

import numpy as np

import torch
import torch.nn as nn

def create_word_vectors(word_embedding_path, save_embedding_path, save_word_idx_path, save_idx_word_path):
    with open(word_embedding_path, 'r') as f:
        embedding = f.readlines()
    embedding = embedding[1:]           # to remove the line containing number of word embeddings and dimensions
    vecs = []
    word_idx = dict()
    idx_word = dict()
    for idx, vec in enumerate(embedding):
        vec = vec.split(' ')
        vecs.append(list(map(float, vec[1:-1])))
        word = vec[0].lower()
        word_idx[word] = idx
        idx_word[idx] = word
    
    word_vectors = torch.tensor(vecs)           # the word vector tensor

    # Save the word vector tensor
    torch.save(os.path.join(word_vectors,'word_vectors.pt'), save_embedding_path)

    # Save the word to idx dictionary
    with open(save_word_idx_path, 'wb') as f:
        pickle.dump(os.path.join(word_idx,'word_idx.pkl'), f)

    # Save the idx to word dictionary
    with open(save_idx_word_path, 'wb') as f:
        pickle.dump(os.path.join(idx_word,'idx_word.pkl'), f)

if __name__ == "__main__":
    word_embedding_path = "/home/aman_khullar/how2/how2-release/word_embedding/cmu_partition.train.vec"
    save_embedding_path = "/home/aman_khullar/how2/temp/"
    save_word_idx_path = "/home/aman_khullar/how2/temp/"
    save_idx_word_path = "/home/aman_khullar/how2/temp/"
    create_word_vectors(word_embedding_path, save_embedding_path, save_word_idx_path, save_idx_word_path)

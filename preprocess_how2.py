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
    torch.save(word_vectors, os.path.join(save_embedding_path, 'word_vectors.pt'))
    print("Saved word vector Tensor")

    # Save the word to idx dictionary
    with open(os.path.join(save_word_idx_path, 'word_idx.pkl'), 'wb') as f:
        pickle.dump(word_idx, f)
    print("Saved Word to Idx dictionary")

    # Save the idx to word dictionary
    with open(os.path.join(save_idx_word_path, 'idx_word.pkl'), 'wb') as f:
        pickle.dump(idx_word, f)
    print("Saved Idx to word dictionary")

if __name__ == "__main__":
    word_embedding_path = "/home/aman_khullar/how2/how2-release/word_embedding/cmu_partition.train.vec"
    save_embedding_path = "/home/aman_khullar/how2/temp/"
    save_word_idx_path = "/home/aman_khullar/how2/temp/"
    save_idx_word_path = "/home/aman_khullar/how2/temp/"
    create_word_vectors(word_embedding_path, save_embedding_path, save_word_idx_path, save_idx_word_path)

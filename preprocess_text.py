import re
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
import numpy as np
import os
import sys
import torch

def get_pretrained_model(glove_input_file):
    word2vec_output_file = glove_input_file + '.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    return word2vec_output_file
    
def get_model(glove_path):
    filename = get_pretrained_model(glove_path)
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model

def preprocess(text, stop_words):
    text = text.lower()
    sentences = sent_tokenize(text)
    tweet_tokenizer = TweetTokenizer()
    document = []
    for sentence in sentences:
        words = tweet_tokenizer.tokenize(sentence)
        doc = [word for word in words if word not in stop_words]
        document.append(doc)
    return document

def document_vector(glove_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in glove_model.vocab]
    return np.mean(glove_model[doc], axis = 0)

def generate_embeddings(path, glove_path, sentence_path):
    num_files = len([item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item)) and '.txt' in item])
    for idx in range(1, num_files):
        lines = []
        with open(path + str(idx) + ".txt") as f:
            for line in f:
                if re.match('\d+:\d+', line) is None:
                    line = line.replace('[MUSIC]', '')
                    lines.append(line.strip())
        text = ' '.join(lines)
        
        stop_words = stopwords.words('english')
        doc = preprocess(text, stop_words)
        model = get_model(glove_path)
        
        embedding_matrix = {}
        for sentence in doc:
            single_sentence_embed = document_vector(model, sentence)
#             x.append(single_sentence_embed)
            sentence_embeddings = np.array(single_sentence_embed)
            sentence_embeddings = torch.from_numpy(sentence_embeddings)
            sentence_list = " ".join(sentence)
            embedding_matrix[sentence_list] = sentence_embeddings
        
        # Save the embedding dictionary for faster loading
        save_path = sentence_path + str(idx) + '.pt'
        torch.save(embedding_matrix, save_path)


        
def download_data():
    """
    This needs to be exectuted if the nltk library data and the stopwords are not installed
    """
    download('punkt') #tokenizer, run once
    download('stopwords') #stopwords dictionary, run once

def main(base_path, glove_path):
    num_courses = 25
    for idx in range(1, num_courses):
        transcript_path = base_path + str(idx) + "/" + "transcripts/"
        sentence_path = base_path + str(idx) + '/' + 'sentence_features/'
        os.system('mkdir ' + sentence_path)
        generate_embeddings(transcript_path, glove_path, sentence_path)

if __name__ == "__main__":
    base_path = '/home/anish17281/NLP_Dataset/dataset/'
    glove_path = '/home/amankhullar/glove_data/glove.6B.300d.txt'
    main(base_path, glove_path)

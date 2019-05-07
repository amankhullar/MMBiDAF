import numpy as np
import torch
from layers.encoding import *
from layers.bidaf import *
import torch.nn as nn

class MMBiDAF(nn.Module):
    """
    The combination of the Bidirectional Attention Flow model and the Multimodal Attention Layer model.

    Follows a high-level structure inspired from the BiDAF implementation by Chris Chute.

        - Embedding layer : Embed the text, audio and the video into suitable embeddings using Glove, MFCC and VGG respectively.
        - Encoder layer : Encode the embedded sequence.
        - Attention Flow layer : Apply the bidirectional attention mechanism for the multimodal data.
        - Modality aware encoding : Encode the modality aware sequence
        - Multimodal Attention : Apply the attention mechanism for the separate modality of data.
        - Ouput layer : Simple Softmax layer to generate the probability distribution over the textual data for extractive summary.
    
    Args:
        word_vectors (torch.Tensor) : Pre-trained word vectors (GLoVE).
        image_vectors (torch.Tensor) : Pre-trained image features (ResNet).
        audio_vectors (torch.Tensor) : Pre-trained audio features (MFCC).
        hidden_size (int) : Number of features in the hidden state at each layer.
        drop_prob (float) : Dropout probability.
    """

    def __init__(self, text_embedding_size, audio_embedding_size=300, hidden_size, drop_prob = 0.):
        super(MMBiDAF, self).__init__()
        self.emb = TextEmbedding(embedding_size=text_embedding_size,
                                 hidden_size=hidden_size,
                                 drop_prob=drop_prob)
        
        self.text_enc = RNNEncoder(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   drop_prob=drop_prob)

        self.audio_encoder = RNNEncoder(input_size=audio_embedding_size, 
                                        hidden_size=hidden_size, 
                                        num_layers=3, 
                                        drop_prob=drop_prob)

        self.image_enc = ImageEncoder()

        self.bidaf_attention = BiDAFAttention(2*hidden_size, 
                                               drop_prob=drop_prob)

        self.text_image_enc = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

        self.text_audio_enc = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)


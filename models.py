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

    def __init__(self, text_embedding_size=300, audio_embedding_size=300, hidden_size, drop_prob = 0.):
        super(MMBiDAF, self).__init__()
        self.emb = TextEmbedding(embedding_size=text_embedding_size,
                                 hidden_size=hidden_size,
                                 drop_prob=drop_prob)
        
        self.text_enc = RNNEncoder(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   drop_prob=drop_prob)

        self.audio_enc = RNNEncoder(input_size=audio_embedding_size, 
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

    def forward(self, embedded_text, original_text_lengths, audio_emb, original_audio_lengths, image_emb):
        text_emb = self.emb(embedded_text)
        text_encoded = self.text_enc(text_emb, original_text_lengths)

        audio_encoded = self.audio_enc(audio_emb, original_audio_lengths)

        image_emb = torch.reshape(image_emb, (-1, image_emb.size(2), image_emb.size(3), image_emb.size(4)))
        image_encoded = self.image_enc(image_emb)
        image_encoded = torch.reshape(image_encoded, (image_encoded.size(0), -1))
        image_linear_layer = nn.Linear(image_encoded.size(-1), 300)
        image_encoded = image_linear_layer(image_encoded)

        


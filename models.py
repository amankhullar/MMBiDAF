import numpy as np
import torch
from layers.encoding import *
from layers.attention import *
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

    def __init__(self, hidden_size, audio_embedding_size, embedding_size=300, drop_prob=0., max_text_length=48):
        super(MMBiDAF, self).__init__()
        self.emb = Embedding(embedding_size=embedding_size,
                             hidden_size=hidden_size,
                             drop_prob=drop_prob)
        
        self.a_emb = Embedding(embedding_size=audio_embedding_size,     # Since audio embedding size is not 300, we need another highway encoder layer
                               hidden_size=hidden_size,                 # and we cannot increase the hidden size beyond 100
                               drop_prob=drop_prob)

        self.text_enc = RNNEncoder(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   drop_prob=drop_prob)

        self.audio_enc = RNNEncoder(input_size=hidden_size, 
                                     hidden_size=hidden_size, 
                                     num_layers=1, 
                                     drop_prob=drop_prob)

        self.image_enc = RNNEncoder(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    drop_prob=drop_prob)

        self.image_keyframes_emb = ImageEmbedding(encoded_image_size=2)

        self.bidaf_att_audio = BiDAFAttention(2*hidden_size, 
                                              drop_prob=drop_prob)

        self.bidaf_att_image = BiDAFAttention(2*hidden_size, 
                                              drop_prob=drop_prob)

        self.mod_t_a = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

        self.mod_t_i = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

        self.multimodal_att_decoder = MultimodalAttentionDecoder(hidden_size,
                                                                 max_text_length,
                                                                 drop_prob)


    def forward(self, embedded_text, original_text_lengths, embedded_audio, original_audio_lengths, transformed_images, original_image_lengths, hidden_gru=None):
        text_emb = self.emb(embedded_text)                                                          # (batch_size, num_sentences, hidden_size)
        text_encoded = self.text_enc(text_emb, original_text_lengths)                               # (batch_size, num_sentences, 2 * hidden_size)

        audio_emb = self.a_emb(embedded_audio)                                                      # (batch_size, num_audio_envelopes, hidden_size)
        audio_encoded = self.audio_enc(audio_emb, original_audio_lengths)                           # (batch_size, num_audio_envelopes, 2 * hidden_size)

        original_image_size = transformed_images.size()                                             # (batch_size, num_keyframes, num_channels, transformed_image_size, transformed_image_size)
        # Combine images across videos in a batch into a single dimension to be embedded by ResNet
        transformed_images = torch.reshape(transformed_images, (-1, transformed_images.size(2), transformed_images.size(3), transformed_images.size(4)))    # (batch_size * num_keyframes, num_channels, transformed_image_size, transformed_image_size)
        image_emb = self.image_keyframes_emb(transformed_images)                                    # (batch_size * num_keyframes, encoded_image_size, encoded_image_size, 2048)
        image_emb = torch.reshape(image_emb, (image_emb.size(0), -1))                               # (batch_size * num_keyframes, encoded_image_size * encoded_image_size * 2048)
        image_linear_layer = nn.Linear(image_emb.size(-1), 300)                                     # Linear layer for linear transformation
        image_emb = image_linear_layer(image_emb)                                                   # (batch_size * num_keyframes, 300)
        image_emb = torch.reshape(image_emb, (original_image_size[0], original_image_size[1], -1))  # (batch_size, num_keyframes, 300)
        image_emb = self.emb(image_emb)                                                             # (batch_size, num_keyframes, hidden_size)
        image_encoded = self.image_enc(image_emb, original_image_lengths)                           # (batch_size, num_keyframes, 2 * hidden_size)

        # TODO: This will only work for batch_size = 1. Add support for larger batches
        text_mask = torch.ones(1, embedded_text.size(1))                                            # (batch_size, padded_seq_length)
        audio_mask = torch.ones(1, embedded_audio.size(1))                                          # (batch_size, padded_seq_length)
        image_mask = torch.ones(1, transformed_images.size(1))                                      # (batch_size, padded_seq_length)

        text_audio_att = self.bidaf_att_audio(text_encoded, audio_encoded, text_mask, audio_mask)   # (batch_size, num_sentences, 8 * hidden_size)
        text_image_att = self.bidaf_att_image(text_encoded, image_encoded, text_mask, image_mask)   # (batch_size, num_sentences, 8 * hidden_size)

        mod_text_audio = self.mod_t_a(text_audio_att, original_text_lengths)                        # (batch_size, num_sentences, 2 * hidden_size)
        mod_text_image = self.mod_t_i(text_image_att, original_text_lengths)                        # (batch_size, num_sentences, 2 * hidden_size)

        if hidden_gru is None:
            mm_att, hidden_gru = self.multimodal_att_decoder(mod_text_audio, mod_text_image)        # (batch_size, num_sentences, )
        else:
            mm_att, hidden_gru = self.multimodal_att_decoder(mod_text_audio, mod_text_image, hidden_gru)

        return mm_att, hidden_gru

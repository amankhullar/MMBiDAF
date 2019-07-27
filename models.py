import sys
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

    def __init__(self, hidden_size, text_embedding_size, audio_embedding_size, image_embedding_size, drop_prob=0., max_transcript_length=405):
        super(MMBiDAF, self).__init__()
        self.emb = Embedding(embedding_size=text_embedding_size,
                             hidden_size=hidden_size,
                             drop_prob=drop_prob)
        
        self.a_emb = Embedding(embedding_size=audio_embedding_size,     # Since audio embedding size is not 300, we need another highway encoder layer
                               hidden_size=hidden_size,                 # and we cannot increase the hidden size beyond 100
                               drop_prob=drop_prob)

        self.i_emb = Embedding(embedding_size=image_embedding_size,     # Since image embedding size is not 300, we need another highway encoder layer
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

        self.image_keyframes_emb = ImageEmbedding()

        self.bidaf_att_audio = BiDAFAttention(2*hidden_size, 
                                              drop_prob=drop_prob)

        self.bidaf_att_image = BiDAFAttention(2*hidden_size, 
                                              drop_prob=drop_prob)

        self.mod_t_a = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,          # changed the number of layers for decoder attention
                                         drop_prob=drop_prob)

        self.mod_t_i = RNNEncoder(input_size=8*hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

        self.multimodal_att_decoder = MultimodalAttentionDecoder(text_embedding_size,
                                                                 hidden_size,
                                                                 max_transcript_length,
                                                                 num_layers=1)


    def get_mask(self, X, X_len):
        X_len = torch.LongTensor(X_len)
        maxlen = X.size(1)
        idx = torch.arange(maxlen).unsqueeze(0).expand(torch.Size(list(X.size())[:2]))
        len_expanded = X_len.unsqueeze(1).expand(torch.Size(list(X.size())[:2]))
        mask = idx < len_expanded
        return mask

    def forward(self, embedded_text, original_text_lengths, embedded_audio, original_audio_lengths, transformed_images, original_image_lengths, batch_target_indices, original_target_len):
        text_emb = self.emb(embedded_text)                                                          # (batch_size, num_sentences, hidden_size)
        text_encoded, _ = self.text_enc(text_emb, original_text_lengths)                               # (batch_size, num_sentences, 2 * hidden_size)

        audio_emb = self.a_emb(embedded_audio)                                                      # (batch_size, num_audio_envelopes, hidden_size)
        audio_encoded, _ = self.audio_enc(audio_emb, original_audio_lengths)                           # (batch_size, num_audio_envelopes, 2 * hidden_size)

        original_images_size = transformed_images.size()                                             # (batch_size, num_keyframes, num_channels, transformed_image_size, transformed_image_size)
        # Combine images across videos in a batch into a single dimension to be embedded by ResNet
        transformed_images = torch.reshape(transformed_images, (-1, transformed_images.size(2), transformed_images.size(3), transformed_images.size(4)))    # (batch_size * num_keyframes, num_channels, transformed_image_size, transformed_image_size)
        image_emb = self.image_keyframes_emb(transformed_images)                                    # (batch_size * num_keyframes, encoded_image_size=1000)
        image_emb = torch.reshape(image_emb, (original_images_size[0], original_images_size[1], -1))  # (batch_size, num_keyframes, 300)
        image_emb = self.i_emb(image_emb)                                                             # (batch_size, num_keyframes, hidden_size)
        image_encoded, _ = self.image_enc(image_emb, original_image_lengths)                           # (batch_size, num_keyframes, 2 * hidden_size)

        text_mask = self.get_mask(embedded_text, original_text_lengths)
        audio_mask = self.get_mask(embedded_audio, original_audio_lengths)
        image_mask = self.get_mask(image_emb, original_image_lengths)

        text_audio_att = self.bidaf_att_audio(text_encoded, audio_encoded, text_mask, audio_mask)   # (batch_size, num_sentences, 8 * hidden_size)
        text_image_att = self.bidaf_att_image(text_encoded, image_encoded, text_mask, image_mask)   # (batch_size, num_sentences, 8 * hidden_size)

        mod_text_audio, text_audio_hidden = self.mod_t_a(text_audio_att, original_text_lengths)                        # (batch_size, num_sentences, 2 * hidden_size
        mod_text_image, text_img_hidden = self.mod_t_i(text_image_att, original_text_lengths)                        # (batch_size, num_sentences, 2 * hidden_size)

        # if hidden_gru is None:
        #     hidden_gru = self.multimodal_att_decoder.initHidden()
        #     hidden_gru, final_out, sentence_dist = self.multimodal_att_decoder(mod_text_audio, mod_text_image, hidden_gru, text_mask)        # (batch_size, num_sentences, )
        # else:
        #     hidden_gru, final_out, sentence_dist = self.multimodal_att_decoder(mod_text_audio, mod_text_image, hidden_gru, text_mask)

        decoder_hidden = (text_audio_hidden.sum(1) + text_img_hidden.sum(1)).unsqueeze(1)           # (batch_size, num_layers*num_dir, hidden_size)
        decoder_hidden = decoder_hidden.transpose(0,1)                                              # To get the decoder input hidden state in required form
        decoder_cell_state = torch.zeros(1, text_emb.size(0), decoder_hidden.size(-1))              # (num_layer*num_dir, batch, hidden_size)

        decoder_input = torch.zeros(text_emb.size(0), 1, embedded_text.size(-1))                    # (batch, num_dir*num_layers, embedding_size)

        # Teacher forcing
        for idx in range(batch_target_indices.size(1)-1):
            out_distributions, decoder_hidden, decoder_cell_state = self.multimodal_att_decoder(decoder_input, decoder_hidden, decoder_cell_state, mod_text_audio, mod_text_image) 
            #TODO loss calculation
            decoder_input = list()
            for i in range(text_emb.size(0)):
                decoder_input.append(embedded_text[i, int(idx)].unsqueeze(0))         # TODO :Pythonic way
            decoder_input = torch.stack(decoder_input)

        print(out_distributions.size())
        sys.exit()              # Debugging purpose
#         print(len(out_distributions))
#         print(out_distributions[0].size())

        return out_distributions

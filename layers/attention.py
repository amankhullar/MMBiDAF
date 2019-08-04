import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiDAFAttention(nn.Module):
    """
    Bidirectional attention computes attention in two directions:
    The text attends to the modality (image/audio) and the modality attends to the text.

    The output of this layer is the concatenation of:
    [text, text2image_attention, text * text2image_attention, text * image2text_attention] or
    [text, text2audio_attention, text * text2audio_attention, text * audio2text_attention]
    based on the modality used.

    This concatenation allows the attention vector at each timestep, along with the embeddings 
    from previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, text_length, 8 * hidden_size)

    Args:
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.text_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.modality_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.text_modality_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.text_weight, self.modality_weight, self.text_modality_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, text, modality, text_mask, modality_mask):
        batch_size, text_length, _ = text.size()
        modality_length = modality.size(1)
        s = self.get_similarity_matrix(text, modality)                     # (batch_size, text_length, modality_length)
        text_mask = text_mask.view(batch_size, text_length, 1)          # (batch_size, text_length, 1)
        modality_mask = modality_mask.view(batch_size, 1, modality_length)    # (batch_size, 1, modality_length)
        s1 = masked_softmax(s, modality_mask, dim=2)                       # (batch_size, text_length, modality_length)
        s2 = masked_softmax(s, text_mask, dim=1)                        # (batch_size, text_length, modality_length)

        # (batch_size, text_length, modality_length) x (batch_size, modality_length, hidden_size) => (batch_size, text_length, hidden_size)
        a = torch.bmm(s1, modality)

        # (batch_size, text_length, text_length) x (batch_size, text_length, hidden_size) => (batch_size, text_length, hidden_size) 
        b = torch.bmm(torch.bmm(s1, s2.transpose(1,2)), text)

        x = torch.cat([text, a, text * a, text * b], dim = 2)            # (batch_size, text_length, 4 * hidden_size)

        return x

    def get_similarity_matrix(self, text, modality):
        """
        Get the "similarity matrix" between text and the modality (image/audio).

        Concatenate the three vectors then project the result with a single weight matrix. This method is more
        memory-efficient implementation of the same operation.

        This is the Equation 1 of the BiDAF paper.
        """
        text_length, modality_length = text.size(1), modality.size(1)
        text = F.dropout(text, self.drop_prob, self.training)           # (batch_size, text_length, hidden_size)
        modality = F.dropout(modality, self.drop_prob, self.training)         # (batch_size, modality_length, hidden_size)

        # Shapes : (batch_size, text_length, modality_length)
        s0 = torch.matmul(text, self.text_weight).expand([-1, -1, modality_length])
        s1 = torch.matmul(modality, self.modality_weight).transpose(1,2).expand([-1, text_length, -1])
        s2 = torch.matmul(text * self.text_modality_weight, modality.transpose(1,2))
        s = s0 + s1 + s2 + self.bias

        return s


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs

class MultimodalAttentionDecoder(nn.Module):
    """
    Multimodal Attention decoder class
    Parameters : 
    text_embedding_size (The Modality layer output) : (batch, max_seq_len, 2*hidden_size)
    hidden_size (The decoder output dimension) : (batch, 1, hidden_size) where hidden size is the that of the decoder
    output_size (The size of the input sentences with padding) : (batch, max_seq_len)
    num_layers (The number of layers of the decoder)
    dropout (The dropout after the decoder)
    """
    def __init__(self, text_embedding_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(MultimodalAttentionDecoder, self).__init__()
        self.text_embedding_size = text_embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # For text-audio attention
        self.W1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.Wc1 = nn.Linear(1, 2 * self.hidden_size)
        self.v1 = nn.Linear(2 * self.hidden_size, 1)
        self.tanh = nn.Tanh()

        # For text-image attention
        self.W3 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.W4 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.Wc2 = nn.Linear(1, 2 * self.hidden_size)
        self.v2 = nn.Linear(2 * self.hidden_size, 1)
        # self.tanh2 = nn.Tanh()

        # For multimodal attention
        self.W_beta_1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)       # the decoder hidden size should be 2 * hidden_size at every timestep (For decoder hidden state)
        self.W_beta_2 = nn.Linear(self.hidden_size, 2 * self.hidden_size)       # Linear layer c1
        self.W_beta_3 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)       # the decoder hidden size should be 2 * hidden_size at every timestep (For decoder hidden state)
        self.W_beta_4 = nn.Linear(self.hidden_size, 2 * self.hidden_size)       # Linear layer c2
        self.v_beta_1 = nn.Linear(2 * self.hidden_size, 1)
        self.v_beta_2 = nn.Linear(2 * self.hidden_size, 1)

        # For the output layer
        self.lstm = nn.LSTM(self.text_embedding_size + 2*self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()

    def forward(self, sent_embed, decoder_hidden, decoder_cell_state, text_audio_enc_out, text_img_enc_out, coverage_vec): # sent_embed : (batch, 1, text_embedding_size); decoder_hidden (batch, num_dir * num_layers, hidden_size)
        # For the text-audio attention
        e1 = self.v1(self.tanh(self.W1(text_audio_enc_out) + self.W2(decoder_hidden) + self.Wc1(coverage_vec))) # (batch, max_seq_len, 1)
        att_weights_1 = F.softmax(e1, dim=1)        # (batch, max_seq_len, 1)
        c1 = att_weights_1 * text_audio_enc_out     # (batch, max_seq_len, 2 * hidden_size)
        c1 = torch.sum(c1, dim=1)                   # (batch, 2 * hidden_size)

        # For the text-image attention
        e2 = self.v2(self.tanh(self.W3(text_img_enc_out) + self.W4(decoder_hidden) + self.Wc2(coverage_vec)))
        att_weights_2 = F.softmax(e2, dim=1)
        c2 = att_weights_2 * text_img_enc_out
        c2 = torch.sum(c2, dim=1)       # (batch, 2 * hidden_size)

        # For the multimodal attention
        # e3 = self.v3(self.tanh(self.W5(c1) + self.W2(c2)))      # (batch, 1)
        # c3 = e3 * c1 + e3 * c2              # (batch, 2 * hidden_size)
        e_beta_1 = self.v_beta_1(self.tanh(self.W_beta_1(c1.unsqueeze(1)) + self.W_beta_2(decoder_hidden)))  # (batch, num_dir*num_layers, 1)
        beta_1 = e_beta_1 * c1          # (batch, num_dir*num_layers, 2 * hidden_size)
        beta_1 = torch.sum(beta_1, dim=1)       # (batch, 2 * hidden_size)

        e_beta_2 = self.v_beta_2(self.tanh(self.W_beta_3(c2.unsqueeze(1)) + self.W_beta_4(decoder_hidden)))  # (batch, num_dir*num_layers, 1)
        beta_2 = e_beta_2 * c2          # (batch, num_layers*num_dir, 2 * hidden_size)
        beta_2 = torch.sum(beta_2, dim=1)   # (batch, 2 * hidden_size)

        c3 = beta_1*c1 + beta_2*c2            # (batch, 2 * hidden_size)
        att_cov_dist = e_beta_1*att_weights_1 + e_beta_2*att_weights_2          # (batch, max_seq_len, 1)
        coverage_vec = coverage_vec + att_cov_dist          # (batch, max_seq_len, 1)
        
        cat_input = torch.cat((c3.unsqueeze(1), sent_embed), dim=2)        # (batch, 1, 2*hidden_size + text_embedding_size)

        decoder_out, (decoder_hidden, decoder_cell_state) = self.lstm(cat_input, (decoder_hidden.transpose(0,1), decoder_cell_state))                       # (batch, 1, hidden_size)
        decoder_out = decoder_out.view(-1, decoder_out.size(-1))    # (batch*1, hidden_size)

        final_out = self.softmax(self.out(decoder_out))       # (batch, max_transcript_len) # TODO use masked softmax

        return final_out, decoder_hidden.transpose(0,1), decoder_cell_state, att_cov_dist, coverage_vec

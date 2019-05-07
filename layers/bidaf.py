import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextImageBiDAFAttention(nn.Module):
    """
    Bidirectional attention computes attention in two directions:
    The text attends to the image and the image attends to the text.
    The output of this layer is the concatenation of :
    [text, text2image_attention, text * text2image_attention, text * image2text_attention].
    This concatenation allows the attention vector at each timestep, along with the embeddings 
    from previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, text_length, 8 * hidden_size)

    Args:
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(TextImageBiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.text_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.image_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.text_image_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.text_weight, self.image_weight, self.text_image_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, text, image, text_mask, image_mask):
        batch_size, text_length, _ = text.size()
        image_length = image.size(1)
        s = self.get_similarity_matrix(text, image)                     # (batch_size, text_length, image_length)
        text_mask = text_mask.view(batch_size, text_length, 1)          # (batch_size, text_length, 1)
        image_mask = image_mask.view(batch_size, 1, image_length)       # (batch_size, 1, image_length)
        s1 = masked_softmax(s, image_mask, dim=2)                       # (batch_size, text_length, image_length)
        s2 = masked_softmax(s, text_mask, dim=1)                        # (batch_size, text_length, image_length)

        # (batch_size, text_length, image_length) x (batch_size, image_length, hidden_size) => (batch_size, text_length, hidden_size)
        a = torch.bmm(s1, image)

        # (batch_size, text_length, text_length) x (batch_size, text_length, hidden_size) => (batch_size, text_length, hidden_size) 
        b = torch.bmm(torch.bmm(s1, s2.transpose(1,2)), text)

        x = torch.cat([text, a, text * a, text * b], dim = 2)            # (batch_size, text_length, 4 * hidden_size)

        return x

    def get_similarity_matrix(self, text, image):
        """
        Get the "similarity matrix" between text and the image.

        Concatenate the three vectors then project the result with a single weight matrix. This method is more
        memory-efficient implementation of the same operation.

        This is the Equation 1 of the BiDAF paper.
        """
        text_length, image_length = text.size(1), image.size(1)
        text = F.dropout(text, self.drop_prob, self.training)           # (batch_size, text_length, hidden_size)
        image = F.dropout(image, self.drop_prob, self.training)         # (batch_size, image_length, hidden_size)

        # Shapes : (batch_size, text_length, image_length)
        s0 = torch.matmul(text, self.text_weight).expand([-1, -1, image_length])
        s1 = torch.matmul(image, self.image_weight).transpose(1,2).expand([-1, text_length, -1])
        s2 = torch.matmul(text * self.text_image_weight, image.transpose(1,2))
        s = s0 + s1 + s2 + self.bias

        return s

class TextAudioBiDAFAttention(nn.Module):
    """
    This class could be incorported in the previous class.
    Bidirectional attention computes attention in two directions:
    The text attends to the audio and the audio attends to the text.
    The output of this layer is the concatenation of :
    [text, text2audio_attention, text * text2audio_attention, text * audio2text_attention].
    This concatenation allows the attention vector at each timestep, along with the embeddings 
    from previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, text_length, 8 * hidden_size)

    Args:
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(TextAudioBiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.text_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.audio_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.text_audio_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.text_weight, self.audio_weight, self.text_audio_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, text, audio, text_mask, audio_mask):
        batch_size, text_length, _ = text.size()
        audio_length = audio.size(1)
        s = self.get_similarity_matrix(text, audio)                     # (batch_size, text_length, audio_length)
        text_mask = text_mask.view(batch_size, text_length, 1)          # (batch_size, text_length, 1)
        audio_mask = audio_mask.view(batch_size, 1, audio_length)       # (batch_size, 1, audio_length)
        s1 = masked_softmax(s, audio_mask, dim=2)                       # (batch_size, text_length, audio_length)
        s2 = masked_softmax(s, text_mask, dim=1)                        # (batch_size, text_length, audio_length)

        # (batch_size, text_length, audio_length) x (batch_size, audio_length, hidden_size) => (batch_size, text_length, hidden_size)
        a = torch.bmm(s1, text)

        # (batch_size, text_length, text_length) x (batch_size, text_length, hidden_size) => (batch_size, text_length, hidden_size) 
        b = torch.bmm(torch.bmm(s1, s2.transpose(1,2)), text)

        x = torch.cat([text, a, text * a, text * b], dim = 2)            # (batch_size, text_length, 4 * hidden_size)

        return x

    def get_similarity_matrix(self, text, audio):
        """
        Get the "similarity matrix" between text and the audio.

        Concatenate the three vectors then project the result with a single weight matrix. This method is more
        memory-efficient implementation of the same operation.

        This is the Equation 1 of the BiDAF paper.
        """
        text_length, audio_length = text.size(1), audio.size(1)
        text = F.dropout(text, self.drop_prob, self.training)           # (batch_size, text_length, hidden_size)
        audio = F.dropout(audio, self.drop_prob, self.training)         # (batch_size, audio_length, hidden_size)

        # Shapes : (batch_size, text_length, audio_length)
        s0 = torch.matmul(text, self.text_weight).expand([-1, -1, audio_length])
        s1 = torch.matmul(audio, self.audio_weight).transpose(1,2).expand([-1, text_length, -1])
        s2 = torch.matmul(text * self.text_audio_weight, audio.transpose(1,2))
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
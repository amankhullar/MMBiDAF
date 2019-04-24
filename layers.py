import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from util import masked_softmax

class TextEmbedding(nn.Module):
    """
    Text Embedding layer used by MMBiDAF.
    This implementation is based on the BiDAF implementation by Chris Chute.

    Args:
        word_vectors (torch.Tensor) : Pre-trained word vectors.
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-in out activations.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(TextEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias = False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class RNNEncoder(nn.Module):
    """
    General-purpose layer for encoding a sequence using a bidirectional RNN.

    This encoding is for the text input data. 
    The encoded output is the RNN's hidden state at each position,
    which has shape (batch_size, seq_len, hidden_size * 2).

    Args:
        input_size (int) : Size of a single timestep in the input (The number of expected features in the input element).
        hidden_size (int) : Size of the RNN hidden state.
        num_layers (int) : Number of layers of RNN cells to use.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_prob = 0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first = True, bidirectional = True,
                           dropout = drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save the original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending = True)
        x = x[sort_idx]    # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first = True)

        # Apply RNN
        x, _ = self.rnn(x) # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first = True, total_length = orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class ImageEncoder(nn.Module):
    """
    This is the encoder layer for the images.

    The reference code has been taken from :
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py

    This is from the paper Show, Attend and Tell.
    """
    def __init__(self, encoded_image_size = 14):
        super(ImageEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # I have used ResNet to extract the features, I could probably experiment with VGG
        resnet = torchvision.models.resnet101(pretrained = True) #Pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we are not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()
    
    def forward(self, images):
        """
        Forward propagation of the set of key frames extracted from the video.

        Args:
            images (torch.Tensor) : The input image with dimension (batch_size, 3, image_size, image_size)
        
        Return:
            Encoded images
        """
        out = self.resnet(images)      # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune = True):
        """
        Allow or prevent the calculation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            fine_tune (bool) : Predicate to allow or prevent the gradient calculation.
        """

        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class AudioEncoder(nn.Module):
    """
    This is the Audio encoding layer which encodes the audio features using BiLSTM.

    The code is inpired from the implementation of the paper Listen, Attend and Spell by Alexander-H-Liu.
    https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/asr.py

    Args:
        enc_type : The encoder architecture available with - VGGBiRNN, BiRNN, RNN.
        sample_rate : Sample rate for each RNN layer, concatenated with _. For each layer,
                      the length of ouput on time dimension will be input/sample_rate.
        sample_style : The down sampling mechanism. concat will concatenate multiplt time steps,
                       according to sample rate into one vector, drop will drop the unsampled timesteps.
        dim : Number of cells for each RNN layer (per direction), concatenated with _.
        dropout : Dropout between each layer, concatenated with _.
        rnn_cell : RNN Cell of all layers.
    """
    def __init__(self, example_input, enc_type, sample_rate, sample_style, dim, dropout, rnn_cell):
        super(AudioEncoder, self).__init__()
        # Setting
        input_dim = example_input.shape[-1]
        self.enc_type = enc_type
        self.vgg = False
        self.dims = [int(v) for v in dim.split('_')]
        self.sample_rate = [int(v) for v in sample_rate.split('_')]
        self.dropout = [float(v) for v in dropout.split('_')]
        self.sample_style = sample_style

        # Parameters checking
        assert len(self.sample_rate)==len(self.dropout), 'Number of layer mismatch'
        assert len(self.dropout)==len(self.dims), 'Number of layer mismatch'
        self.num_layers = len(self.sample_rate)
        assert self.num_layers>=1,'AudioEncoder should have at least 1 layer'

        # Construct AudioEncoder
        if 'VGG' in enc_type:
            self.vgg = True
            self.vgg_extractor = VGGExtractor(example_input)
            input_dim = self.vgg_extractor.out_dim

        for l in range(self.num_layers):
            out_dim = self.dims[l]
            sr = self.sample_rate[l]
            drop = self.dropout[l]

            
            if "BiRNN" in enc_type:
                setattr(self, 'layer'+str(l), RNNLayer(input_dim,out_dim, sr, rnn_cell=rnn_cell, dropout_rate=drop,
                                                       bidir=True,sample_style=sample_style))
            elif "RNN" in enc_type:
                setattr(self, 'layer'+str(l), RNNLayer(input_dim,out_dim, sr, rnn_cell=rnn_cell, dropout_rate=drop,
                                                       bidir=False,sample_style=sample_style))
            else:
                raise ValueError('Unsupported Encoder Type: '+enc_type)

            # RNN ouput dim = default output dim x direction x sample rate
            rnn_out_dim = out_dim*max(1,2*('Bi' in enc_type))*max(1,sr*('concat'== sample_style)) 
            setattr(self, 'proj'+str(l),nn.Linear(rnn_out_dim,rnn_out_dim))
            input_dim = rnn_out_dim

    
    def forward(self,input_x,enc_len):
        if self.vgg:
            input_x,enc_len = self.vgg_extractor(input_x,enc_len)
        for l in range(self.num_layers):
            input_x, _,enc_len = getattr(self,'layer'+str(l))(input_x,state_len=enc_len, pack_input=True)
            input_x = torch.tanh(getattr(self,'proj'+str(l))(input_x))
        return input_x,enc_len

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
        a = torch.bmm(s1, text)

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
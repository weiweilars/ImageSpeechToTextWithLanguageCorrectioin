import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import sqrt, cos, sin
import numpy as np
from numpy import inf


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input
    Args:
       n_feats : number of features in the cnn 
    """
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """Residual Convolutional Network

    Args:
       in_channels: input channels
       out_channels: output channels
       kernel: kernel size 
       stride:
       dropout: 
       n_feats: number of features 
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        #pdb.set_trace()
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    '''Bidirectional Gate Recurrent Network 
    
    Args:
       rnn_dim : input dimension 
       hidden_size: hidden dimension 
       dropout
       batch_first : batch size is the first dimension 
    '''
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x



class MultiheadAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super(MultiheadAtten, self).__init__()

        assert embed_dim % num_heads == 0

        self.dropout = dropout 

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        self.out_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.empty(embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        
        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.in_proj_weight)
            
        nn.init.kaiming_uniform_(self.out_proj_weight, a=sqrt(5)) # better than xavier_uniform_
        
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias,0.)
            nn.init.constant_(self.out_proj_bias,0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
            
        tgt_len, bsz, embed_dim = query.size()
            
        scaling = float(self.head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            _b = self.in_proj_bias
            _start = 0
            _end = self.embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b_1 = _b[_start:_end]
            else:
                _b_1 = None
            q = F.linear(query, _w, _b_1)

            _w = self.in_proj_weight[_end:, :]
            if _b is not None:
                _b_2 = _b[_end:]
            else:
                _b_2 = None
            k, v = F.linear(key, _w, _b_2).chunk(2, dim=-1)

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0,1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1,2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

            
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
            
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            
        return attn_output, attn_output_weights
            
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim=2048,
                 dropout=0.1,
                 activation="relu"):

        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        self.self_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)
        self.ff_linear1 = Linear(hid_dim, pf_dim, w_init_gain=activation)
        self.ff_linear2 = Linear(pf_dim, hid_dim)

        self.ff_norm1 = nn.LayerNorm(hid_dim)
        self.ff_norm2 = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_attn_mask=None, src_key_padding_mask=None):
        src2, src_align = self.self_attn(src, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src = self.ff_norm1(src + self.dropout(src2))

        src2 = self.ff_linear2(self.dropout(F.relu(self.ff_linear1(src))))
        src = self.ff_norm2(src + self.dropout(src2))

        return src, src_align

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        # self.cross_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        self.self_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)
        self.cross_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)

        self.ff_linear1 = Linear(hid_dim, pf_dim, w_init_gain=activation)
        self.ff_linear2 = Linear(pf_dim, hid_dim)

        self.ff_norm1 = nn.LayerNorm(hid_dim)
        self.ff_norm2 = nn.LayerNorm(hid_dim)
        self.ff_norm3 = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, tgt_attn_mask=None, src_attn_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None):      
        tgt2, tgt_align = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.ff_norm1(tgt + self.dropout(tgt2))

        tgt2, tgt_src_align = self.cross_attn(tgt, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        tgt = self.ff_norm2(tgt + self.dropout(tgt2))

        tgt2 = self.ff_linear2(self.dropout(F.relu(self.ff_linear1(tgt))))
        tgt = self.ff_norm3(tgt + self.dropout(tgt2))

        return tgt, tgt_align, tgt_src_align
        



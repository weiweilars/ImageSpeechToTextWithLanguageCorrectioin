import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from layers import CNNLayerNorm, ResidualCNN, BidirectionalGRU, EncoderLayer

class ToLetter(nn.Module):
    def __init__(self, params, init_weights=True):
        super(ToLetter, self).__init__()

        n_cnn_layers = params['n_cnn_layers']
        n_rnn_layers = params['n_rnn_layers']
        rnn_dim = params['rnn_dim']
        n_class = params['n_class']
        n_feats = params['n_features']
        stride = params['stride']
        dropout = params['dropout']
        n_feats = n_feats//2

        
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2) 

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        # n_feats is the number of feature map
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(rnn_dim, n_class)

        self.type = type
        self.device = device
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) 
        x = x.transpose(1, 2) 
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x_mid = self.linear(x)
        x = self.classifier(x_mid)
        return x, x_mid

    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

    
class post_model(nn.Module):
    pass

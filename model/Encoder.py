import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from torch.nn import  LayerNorm

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=7):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[ :, 0::2 ] = torch.sin(position * div_term)
        pe[ :, 1::2 ] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[ :, :x.size(1) ],
                         requires_grad=False)
        
        return self.dropout(x)
        #return x

class TransformerEncoder(nn.Module):
    
    def __init__(self,d_model=256):
        super(TransformerEncoder, self).__init__()
        #self.cls_token=nn.Parameter(torch.zeros(1, 1,d_model))
        self.d_model=d_model
        encoder_norm = LayerNorm(d_model)
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8,dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,norm=encoder_norm)
        self.pos_emb=PositionalEncoding(d_model)
        
    def forward(self,x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1)
        x=x*math.sqrt(self.d_model)
        x=self.pos_emb(x)
        x=x.permute(1, 0, 2)
        output=self.transformer_encoder(x)
        output=output.permute(1, 0, 2)
        return output

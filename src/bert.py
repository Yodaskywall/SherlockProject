import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import Encoder
import pickle


class Bert(nn.Module):

    def __init__(self, vocab_size, n_embd, n_heads, max_length, n_layers, dropout, device=None):
        super().__init__()
        self.encoder = Encoder(vocab_size, n_heads,
                         n_embd, max_length, n_layers, dropout, device=device)
        self.compress = nn.Linear(n_embd, 1, device=device)
        self.device = device
        self.max_length = max_length
        self.n_embd = n_embd
    
    def forward(self, sentences, segments, targets=None):
        x = self.encoder(sentences, segments)
        logits = self.compress(x[:, :3, :]).view(-1, 3)
        
        if targets is None:
            loss = None

        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


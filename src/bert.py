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
        self.un_embed = nn.Linear(n_embd, vocab_size, device=device)
        self.device = device
        self.max_length = max_length
        self.n_embd = n_embd
        self.vocab_size = vocab_size
    
    def forward(self, sentences, segments, mask, targets=None):
        """
        sentences -> (B, T)
        segments -> (B, T)
        mask -> (B, T)
        targets -> (MT, 1)
        """
        x = self.encoder(sentences, segments)
        logits = self.un_embed(x)

        if targets is None:
            loss = None

        else:
            # logits -> (B, T, vocab_size)
            relevant_logits = logits.view(-1,  self.vocab_size)[mask.view(-1).nonzero()].squeeze()
            targets = targets.squeeze()
            loss = F.cross_entropy(relevant_logits, targets)

        return logits, loss


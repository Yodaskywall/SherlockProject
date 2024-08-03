import torch
import torch.nn as nn
from torch.nn import functional as F
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding(nn.Module):
    """ Receives tokenised sentences (already with SEPS and CLS) and embeds them """
    def __init__(self, vocab_size, n_embd, max_length):
        super().__init__()
        self.token_embd_mat = nn.Embedding(vocab_size, n_embd)
        self.position_embd_mat = nn.Embedding(max_length, n_embd)
        self.sentence_embd_mat = nn.Embedding(2, n_embd)
        self.max_length


    def forward(self, tokens, segment):
        """ Forward pass behaviour """
        # tokens shape is (B, T)
        # segment is (B, T) containing 1s and 0s indicating if the token belongs to sentece A or B
        # [CLS] + sentence A + [SEP] + sentence B + [SEP]. CLS and first SEP would be sentence A
        token_embd = self.token_embd_mat(tokens) # (B, T, C)
        position_embd = self.position_embd_mat(torch.arange(self.max_length, device=DEVICE)) # (T, C)
        sentence_embd = self.sentence_embd_mat(segment) # (B, T, C)

        embd = token_embd + position_embd + sentence_embd # Broadcast occurs for position embeddings
        return embd


class Attention(nn.Module):

    def __init__(self, n_heads, n_embd, dropout):
        super().__init__()
        self.key_mat = nn.Linear(n_embd, n_embd, bias=False)
        self.query_mat = nn.Linear(n_embd, n_embd, bias=False)
        self.value_mat = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        assert n_embd % n_heads == 0
        self.n_heads = n_heads


    def forward(self, x):
        # X is (B, T, C) C = n_embd
        B, T, C = x.size()

        keys = self.key_mat(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nH, T, hS)
        queries = self.query_mat(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nH, T, hS)
        values = self.value_mat(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nH, T, hS)

        # NOTE: must do it with transpose above, because we split the last dimension into 2, then move dims

        # Calculates affinities
        aff = queries @ keys.transpose(-2, -1) # (B, nH, T, hS) @ (B, nH, hS, T) -> (B, hs, T, T)
        aff *= (1.0 / math.sqrt(keys.size(-1))) # Rescaling
        aff = F.softmax(aff, dim=-1)
        aff = self.dropout(aff)

        out = aff @ values # (B, hs, T, T) @ (B, nH, T, hS)) -> (B, nH, T, hS)
        out = out.transpose(1,2).contiguous().view(B, T, C) # (B, nH, T, hs) -> (B, T, nH, hs) -> (B, T, C)
        # Don't really understand the contiguous

        out = self.dropout2(self.proj(out)) # We project, because we don't just want to concatenate, we mix the data into new embds
        return out

class FFN(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.lin1 = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, T, C)
        x = self.lin1(x) # (B, T, 4C)
        x = self.act(x) # (B, T, 4C)
        x = self.lin2(x) # (B, T, C) (Projection)
        x = self.dropout(x)
        
        return x


class Block(nn.Module):
    def __init__(self, n_heads, n_embd, dropout):
        super().__init__()
        self.attention = Attention(n_heads, n_embd, dropout)
        self.ffn = FFN(n_embd, dropout)
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x is (B, T, C)

        # NOTE: We deviate from Attention is all you need, performing layern norm before each thing, because
        #       Andrej Karpathy said it's better in his video

        x = self.attention(self.layerNorm1(x)) + x
        x = self.ffn(self.layerNorm2(x)) + x

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embd, max_length, n_layers, dropout):
        self.embedding = Embedding(vocab_size, n_embd, max_length)
        self.blocks = [Block(n_heads, n_embd, dropout) for i in range(n_layers)]
        self.layerNorm = nn.LayerNorm(n_embd)


    def forward(self, tokens, segment):
        embds = self.embedding(tokens, segment)

        for block in self.blocks:
            x = self.block(x)

        x = self.layerNorm(x)
        return x

        



        

import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import Encoder
from tokenizer import Tokenizer
import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Bert(nn.Module):

    N_EMBD = 600
    N_HEADS = 6
    MAX_LENGTH = 500
    N_LAYERS = 6
    DROPOUT = 0.2


    def __init__(self, tokenizer_file):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.tokenizer.load_model(tokenizer_file)
        self.encoder = Encoder(self.tokenizer.vocab_size, self.N_HEADS,
                         self.N_EMBD, self.MAX_LENGTH, self.N_LAYERS, self.DROPOUT, device=DEVICE)
        self.compress = nn.Linear(self.N_EMBD, 1)
    
    def forward(self, sentences, segments):
        sentences, segments = self.pad(sentences, segments)
        x = self.encoder(sentences, segments)
        y = self.compress(x[:, :3, :]).squeeze()
        print(y)


    def pad(self, sentences, segments):
        sentences = torch.stack([F.pad(torch.tensor(sentence), (0,self.MAX_LENGTH-len(sentence)),
                              value=self.tokenizer.pad_token) for sentence in sentences])
        segments = torch.stack([F.pad(torch.tensor(segment), (0,self.MAX_LENGTH-len(segment)),
                              value=0) for segment in segments])
        sentences, segments = sentences.to(DEVICE), segments.to(DEVICE)
        return (sentences, segments)

if __name__ == "__main__":
    bert = Bert("data/tokeniser_data/tokenizer_data.pickle")
    bert = bert.to(DEVICE)
    #bert = bert.to(DEVICE)
    with open("data/processed_train.pickle", "rb") as file:
        sentences, segments, targets = pickle.load(file)

    targets = torch.tensor(targets)

    bert.forward([sentences[0]], [segments[0]])
    

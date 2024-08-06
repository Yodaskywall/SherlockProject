import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import Encoder
import pickle
from bert import Bert
from tokenizer import Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# MODEL HYPERPARAMETERS
N_EMBD = 700
N_HEADS = 7
MAX_LENGTH = 300
N_LAYERS = 6
DROPOUT = 0.2

# TRAINING HYPERPARAMETERS
LR = 5e-05
TRAINING_STEPS = 8000
EVAL_INTERVAL = 500
BATCH_SIZE = 32
EVAL_ITERS = 150

class Classifier(nn.Module):
    def __init__(self, tokenizer, n_embd, n_heads, max_length, n_layers, dropout, device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = Bert(tokenizer.vocab_size, n_embd, n_heads, max_length, n_layers, dropout, device)
        self.bert.load_state_dict(torch.load("saved_models/model_dict_save1"))
        self.bert = self.bert.to(device)

        self.act = nn.GELU()
        self.layerNorm1 = nn.LayerNorm(tokenizer.vocab_size, device=device)
        self.lin1 = nn.Linear(tokenizer.vocab_size, 64, device=device)
        self.layerNorm2 = nn.LayerNorm(64, device=device)
        self.lin2 = nn.Linear(64, 1, device=device)
        



    def forward(self, sentences, segments, targets=None):
        x, _ = self.bert(sentences, segments)
        x = x[:, :3, :]
        x = self.act(x)
        x = self.lin1(self.layerNorm1(x))
        x = self.act(x)
        logits = self.lin2(self.layerNorm2(x)).view(-1, 3)

        if targets is None:
            loss = False

        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def pad(tokenizer, sentences, segments):
        sentences = torch.stack([F.pad(torch.tensor(sentence, dtype=torch.long), (0,MAX_LENGTH-len(sentence)),
                              value=tokenizer.pad_token) for sentence in sentences])
        segments = torch.stack([F.pad(torch.tensor(segment, dtype=torch.long), (0,MAX_LENGTH-len(segment)),
                              value=0) for segment in segments])
        sentences, segments = sentences.to(DEVICE), segments.to(DEVICE)
        return (sentences, segments)  

tokenizer = Tokenizer()
tokenizer.load_model("data/tokeniser_data/tokenizer_data.pickle")
model = Classifier(tokenizer, N_EMBD, N_HEADS, MAX_LENGTH, N_LAYERS, DROPOUT, device=DEVICE)


# Load data
with open("data/processed_train.pickle", "rb") as file:
    sentences, segments, targets = pickle.load(file)


sentences, segments = pad(tokenizer, sentences, segments)
targets = torch.tensor(targets, device=DEVICE, dtype=torch.long)

# Train-validation split
n = int(0.9*len(sentences))
train_sent, train_seg, train_targ = sentences[:n], segments[:n], targets[:n]
val_sent, val_seg, val_targ = sentences[n:], segments[n:], targets[n:]

def get_batch(data):
    sentences, segments, targets = data
    ix = torch.randint(len(sentences) - 1, (BATCH_SIZE,))
    sent_batch, seg_batch, targ_batch = sentences[ix], segments[ix], targets[ix]
    return (sent_batch, seg_batch, targ_batch)

def estimate_loss(eval_iters=EVAL_ITERS):
    out = {}
    splits = {"train": (train_sent, train_seg, train_targ), "val" : (val_sent, val_seg, val_targ)}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if k % 10 == 0:
                print(f"evaluating {k}%...")
            sents, segs, targs  = get_batch(splits[split])
            logits, loss = model(sents, segs, targs)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for iter in range(TRAINING_STEPS):
    # Estimate loss and print it
    if iter % 50 == 0:
        print(f"Iter {iter}")
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch
    sent_batch, seg_batch, targ_batch = get_batch((train_sent, train_seg, train_targ))

    # Evaluate loss
    logits, loss = model(sent_batch, seg_batch, targ_batch)
    # Set gradients to zero before backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(800)
print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save(model.state_dict(), "finetuned_model")
print("Saved Model")

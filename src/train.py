from bert import Bert
import matplotlib.pyplot as plt
import torch
import pickle
import random
from tokenizer import Tokenizer
from torch.nn import functional as F


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


  

# Initialize model
tokenizer = Tokenizer()
tokenizer.load_model("data/tokeniser_data/tokenizer_data.pickle")
model = Bert(tokenizer.vocab_size, N_EMBD, N_HEADS, MAX_LENGTH, N_LAYERS, DROPOUT, device=DEVICE)
model.load_state_dict(torch.load("saved_models/model_dict_save1"))
print(DEVICE)
print(sum(p.numel() for p in model.parameters()))
model = model.to(DEVICE)

# Load data
with open("data/processed_train.pickle", "rb") as file:
    sentences, segments, targets = pickle.load(file)

max_len = 0
for line in sentences:
    if len(line) > max_len:
        max_len = len(line)
print(f"Max len {max_len}")

# Pads and tensorizes data

def pad(tokenizer, sentences, segments):
        sentences = torch.stack([F.pad(torch.tensor(sentence, dtype=torch.long), (0,MAX_LENGTH-len(sentence)),
                              value=tokenizer.pad_token) for sentence in sentences])
        segments = torch.stack([F.pad(torch.tensor(segment, dtype=torch.long), (0,MAX_LENGTH-len(segment)),
                              value=0) for segment in segments])
        sentences, segments = sentences.to(DEVICE), segments.to(DEVICE)
        return (sentences, segments)  

sentences, segments = pad(tokenizer, sentences, segments)
targets = torch.tensor(targets, device=DEVICE, dtype=torch.long)

# Train-validation split
n = int(0.9*len(sentences))
train_sent, train_seg, train_targ = sentences[:n], segments[:n], targets[:n]
val_sent, val_seg, val_targ = sentences[n:], segments[n:], targets[n:]


def generate_MLM_data(sentences, tokenizer):
    # How many tokens to mask?
    n_masks = int(MAX_LENGTH * 0.1)
    n_rand = int(MAX_LENGTH * 0.01)
    n_unch = n_rand

    mask_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_masks))
    rand_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_rand))
    unch_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_unch))
    mask = torch.zeros_like(sentences)
    rand = torch.zeros_like(sentences)
    unch = torch.zeros_like(sentences)
    # Sets ones in the mask in the generated indicies
    for i in range(sentences.shape[0]):
        mask[i, mask_ix[i]] = 1
        rand[i, rand_ix[i]] = 1
        unch[i, unch_ix[i]] = 1
    # Only masks non-special tokens
    not_special_tokens = ((sentences != tokenizer.cls_token) & (sentences != tokenizer.sep_token)) & (sentences != tokenizer.pad_token)
    mask = mask & not_special_tokens
    rand = rand & not_special_tokens
    unch = unch & not_special_tokens

    total_mask = (mask | rand) | unch

    # Flat array of targets (replaced tokens)
    targets = sentences.view(-1)[total_mask.view(-1).nonzero()]
    # Applies mask
    sentences = torch.where(mask == 1, tokenizer.mask_token, sentences)
    sentences = torch.where(rand == 1, torch.randint_like(sentences, 0, tokenizer.vocab_size), sentences)
    return sentences, targets, total_mask


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
            sents, targs, masks = generate_MLM_data(sents, tokenizer)
            logits, loss = model(sents, segs, masks, targs)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# TRAINING LOOP

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
    sent_batch, targ_batch, mask_batch = generate_MLM_data(sent_batch, tokenizer)

    # Evaluate loss
    logits, loss = model(sent_batch, seg_batch, mask_batch, targ_batch)
    # Set gradients to zero before backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(800)
print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save(model.state_dict(), "model_dict_save")
print("Saved Model")

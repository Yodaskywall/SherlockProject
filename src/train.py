from bert import Bert
import matplotlib.pyplot as plt
import torch
import pickle
import math
import random
from tokenizer import Tokenizer
from torch.nn import functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# MODEL HYPERPARAMETERS
N_EMBD = 800
N_HEADS = 8
MAX_LENGTH = 300
N_LAYERS = 6
DROPOUT = 0.2

# TRAINING HYPERPARAMETERS
LR = 1e-05
TRAINING_STEPS = 8000
EVAL_INTERVAL = 500
BATCH_SIZE = 32
EVAL_ITERS = 150


  

# Initialize model
tokenizer = Tokenizer()
model = Bert(tokenizer.vocab_size, N_EMBD, N_HEADS, MAX_LENGTH, N_LAYERS, DROPOUT, device=DEVICE)
print(DEVICE)
print(sum(p.numel() for p in model.parameters()))
model = model.to(DEVICE)

# Load data
with open("data/allbooks.pickle", "rb") as file:
    text = pickle.load(file)

text = torch.tensor(text)

# Train-validation split
n = int(0.9*len(text))
train_text = text[:n]
val_text = text[n:]


def generate_MLM_data(sentences, tokenizer):
    # How many tokens to mask?
    n_masks = int(MAX_LENGTH * 0.11)
    n_rand = int(MAX_LENGTH * 0.01)
    n_unch = n_rand

    # Generates masked indicies
    mask_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_masks))
    rand_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_rand))
    unch_ix = torch.randint(MAX_LENGTH - 1, (sentences.shape[0], n_unch))

    # Initializes masks
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

    # Combines all types of mask
    total_mask = (mask | rand) | unch

    # Flat array of targets (replaced tokens)
    targets = sentences.view(-1)[total_mask.view(-1).nonzero()]
    
    # Applies masks
    sentences = torch.where(mask == 1, tokenizer.mask_token, sentences)
    sentences = torch.where(rand == 1, torch.randint_like(sentences, 0, tokenizer.vocab_size), sentences)
    
    return sentences, targets, total_mask


def get_batch(text):
    MEAN = 60
    STD = 16

    # Generates random sentence lengths
    sentence_lengths = torch.round(torch.abs((STD * torch.randn(BATCH_SIZE)) + MEAN)).long()
    max_batch_length = torch.max(sentence_lengths).item()

    # Generates random indicies
    ixs = torch.randint(len(text) - max_batch_length, (BATCH_SIZE,))

    # Fetches sentences, and pads them
    sentences = torch.stack([F.pad(text[ ix : ix + sentence_lengths[i] ], (0, MAX_LENGTH - sentence_lengths[i]),
                              value=tokenizer.pad_token) for i,ix in enumerate(ixs)])

    # Creates all-zeros segments
    # Not sure segments are needed for pre-training phase
    segments = torch.zeros_like(sentences, device=DEVICE)
    sentences = sentences.to(DEVICE)

    return sentences, segments

def estimate_loss(eval_iters=EVAL_ITERS):
    out = {}
    splits = {"train": train_text, "val" : val_text}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if k % 10 == 0:
                print(f"evaluating {k}/{eval_iters}...")
            sents, segs  = get_batch(splits[split])
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
    sent_batch, seg_batch = get_batch(train_text)
    sent_batch, targ_batch, mask_batch = generate_MLM_data(sent_batch, tokenizer)

    # Evaluate loss
    logits, loss = model(sent_batch, seg_batch, mask_batch, targ_batch)

    # Set gradients to zero before backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(500)
print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save(model.state_dict(), "model_dict_save")
print("Saved Model")

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


tokenizer = Tokenizer()
tokenizer.load_model("data/tokeniser_data/tokenizer_data.pickle")
model = Bert(tokenizer.vocab_size, N_EMBD, N_HEADS, MAX_LENGTH, N_LAYERS, DROPOUT, device=DEVICE)
model.load_state_dict(torch.load("saved_models/model_dict_save2"))
model.to(DEVICE)

def pad(tokenizer, sentence, segment, mask):
        sentence = F.pad(torch.tensor(sentence, dtype=torch.long), (0,MAX_LENGTH-len(sentence)),
                              value=tokenizer.pad_token)
        segment = F.pad(torch.tensor(segment, dtype=torch.long), (0,MAX_LENGTH-len(segment)),
                              value=0)
        mask = F.pad(torch.tensor(mask, dtype=torch.long), (0,MAX_LENGTH-len(mask)),
                              value=0)
        sentence, segment, mask = sentence.to(DEVICE), segment.to(DEVICE), mask.to(DEVICE)
        return (torch.unsqueeze(sentence, 0), torch.unsqueeze(segment, 0), torch.unsqueeze(mask, 0))  

def encode_fragments(fragments, tokenizer):
    segment_value = 0
    segment = 3*[0]
    mask = 3*[0]
    sentence = 3*[tokenizer.cls_token]
    for fragment in fragments:
        if fragment == 0:
            sentence.append(tokenizer.mask_token)
            mask.append(1)
            segment.append(segment_value)
        elif fragment == 1:
            sentence.append(tokenizer.sep_token)
            segment.append(segment_value)
            segment_value = 1
            mask.append(0)
        else:
            encoded = tokenizer.encode(fragment)
            sentence += encoded
            segment += len(encoded)*[segment_value]
            mask += len(encoded)*[0]
    return pad(tokenizer, sentence, segment, mask)

sentence, segment, mask = encode_fragments(["Me encantan las frutas verdes, además de las verduras", 1, "Me encanta cuando las", 0, " son verdes", 1], tokenizer)
model.eval()
logits, _ = model(sentence, segment, mask)
relevant_logits = logits.view(-1,  tokenizer.vocab_size)[mask.view(-1).nonzero()].view(-1, tokenizer.vocab_size)
probs = F.softmax(relevant_logits, 1)
generated_tokens = torch.multinomial(probs, num_samples=1)
text = tokenizer.decode([generated_tokens[i].item() for i in range(len(generated_tokens))])

print("Me encantan las frutas verdes, además de las verduras Me encanta cuando las" + text + " son verdes")



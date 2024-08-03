import regex as re
import pickle

class Tokenizer:
    """ Tokenizer that works with BPE, first dividing text according to a regex pattern """
    REG_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    
    def __init__(self):
        self.pattern = re.compile(self.REG_PAT)
        self.vocab_size = 0
        self.merges = {}
        self.vocab = {}


    def train(self, text, vocab_size, verbose=False, save=False):
        assert vocab_size >= 256
        self.vocab_size = vocab_size
        n_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern, text)

        token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        self.merges = {}
        self.vocab = { idx : bytes([idx]) for idx in range(256)}

        for i in range(n_merges):
            stats = self.get_stats(token_chunks)
            most_repeated = max(stats, key=stats.get)
            most_reps = stats[most_repeated]

            if most_reps == 0:
                break
            
            idx = 256 + i
            
            self.merges[most_repeated] = idx
            self.vocab[idx] = self.vocab[most_repeated[0]] + self.vocab[most_repeated[1]]

            token_chunks = self.replace(token_chunks, most_repeated, idx)
            
            if verbose:        
                print(f"Merging {self.vocab[idx]} ({most_repeated}) into new token {idx} (repeated {most_reps} times)")

        tokens = [token for chunk in token_chunks for token in chunk]

        if save:
            with open("data/tokeniser_data/encoded_text.pickle", "wb") as file:
                pickle.dump(tokens, file, protocol=pickle.DEFAULT_PROTOCOL)

            with open("data/tokeniser_data/tokenizer_data.pickle", "wb") as file:
                pickle.dump((self.merges, self.vocab), file, protocol=pickle.DEFAULT_PROTOCOL)

        return self.merges, self.vocab, tokens


    def encode(self, text):
        chunks = re.findall(self.pattern, text)

        encoded_token_chunks = [self.encode_chunk(chunk) for chunk in chunks]
        tokens = [token for chunk in encoded_token_chunks for token in chunk]
        
        return tokens

    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8")
        return text


    def load_model(self, filename):
        with open(filename, "rb") as file:
            self.merges, self.vocab = pickle.load(file)


    # ----------------------------------------------------------
    # Auxiliar methods

    def encode_chunk(self, text_chunk):

        tokens = list(text_chunk.encode("utf-8"))
   
        while len(tokens) > 1:
            stats = self.get_stats([tokens])
            pair = min(stats, key=lambda p : self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.replace_in_chunk(tokens, pair, idx)       
        
        return tokens

    def get_stats(self, token_chunks):
        stats = {}
        for chunk in token_chunks:
            for pair in zip(chunk, chunk[1:]):
                stats[pair] = stats.get(pair, 0) + 1
        return stats

    def replace_in_chunk(self, chunk, pair, new):
        new_tokens = []
        i = 0

        while (i < len(chunk)):

            if (i < len(chunk) -1 and chunk[i] == pair[0] and chunk[i+1] == pair[1]):
                new_tokens.append(new)
                i+=2
            else:

                new_tokens.append(chunk[i])
                i+=1
        return new_tokens

    def replace(self, token_chunks, pair, new):
        new_chunks = []
        for chunk in token_chunks:
            new_chunks.append(self.replace_in_chunk(chunk, pair, new))
        return new_chunks



if __name__ == "__main__":
    with open("data/train.csv", encoding="utf-8") as file:
        text = file.read()

    tok = Tokenizer()
    merges, vocab, encoding = tok.train(text, 5000, True, True)

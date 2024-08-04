from tokenizer import Tokenizer
import pickle
import csv

def process_data(in_filename, out_filename, test=False):
    tok = Tokenizer()
    tok.set_special_tokens(5000) # DELETE THIS LINE IF RETRAINED
    tok.load_model("data/tokeniser_data/tokenizer_data.pickle")
    with open(in_filename, encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        encoded_sentences = []
        segments = []
        targets = []
        for line in reader:
            encoded_sent1 = tok.encode(line[1])
            encoded_sent2 = tok.encode(line[2])
            encoded_sentences.append(3*[tok.cls_token] + encoded_sent1 + [tok.sep_token] + encoded_sent2 + [tok.sep_token])
            segments.append([0 for i in range(len(encoded_sent1) + 4)] + [1 for i in range(len(encoded_sent2) + 1)])
            if not test:
                targets.append(int(line[5]))
    
    if test:
        to_dump = (encoded_sentences, segments)
    else:
        to_dump = (encoded_sentences, segments, targets)

    with open(out_filename, "wb") as file:
        pickle.dump(to_dump, file, protocol=pickle.DEFAULT_PROTOCOL)

process_data("data/test.csv", "data/processed_test.pickle", test=True)
print("TEST FINISHED")
process_data("data/train.csv", "data/processed_train.pickle")
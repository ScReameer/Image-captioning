import torch
import pandas as pd
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer

class Vocabulary:
    def __init__(self, path: list, freq_threshold=3):
        self.tokenizer = get_tokenizer("basic_english")
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.df = pd.read_csv(path)
        self.df['caption'] = self.df['caption'].apply(self.tokenizer)
        self._build()

    def __len__(self):
        return len(self.idx2word)

    def _build(self):
        vocab_dict = {}
        start_idx = 4
        for sentence in self.df['caption'].tolist():
            for word in sentence:
                if word not in vocab_dict.keys():
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
                if vocab_dict[word] == self.freq_threshold:
                    self.word2idx[word] = start_idx
                    self.idx2word[start_idx] = word
                    start_idx += 1
    
    def numericalize(self, text):
        lower_text = self.tokenizer(text)
        tokenized_text = [self.word2idx['<SOS>']] + [
            self.word2idx[token] if token in self.word2idx else self.word2idx["<UNK>"]
            for token in lower_text
        ] + [self.word2idx['<EOS>']]
        return torch.tensor(tokenized_text)
import torch
import pandas as pd
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer

class Vocabulary:
    def __init__(self, text: pd.Series, freq_threshold=5):
        """Creates vocabulary for corpus of text

        Args:
            `text` (`pd.Series`): column from dataframe with captions
            `freq_threshold` (`int`, optional): write words in vocabulary if word frequency >= `freq_threshold`, else word becomes `<UNK>`. Defaults to `5`.
        """
        self.tokenizer = get_tokenizer("basic_english")
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.text: pd.Series = text.copy().astype(str).apply(self.tokenizer)
        self._build()
        del self.text

    def __len__(self):
        return len(self.idx2word)

    def _build(self):
        """Builds vocabulary"""
        vocab_dict = {}
        start_idx = 4
        for sentence in self.text.tolist():
            for word in sentence:
                if word not in vocab_dict.keys():
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
                if vocab_dict[word] == self.freq_threshold:
                    self.word2idx[word] = start_idx
                    self.idx2word[start_idx] = word
                    start_idx += 1
    
    def numericalize(self, text: str):
        """Processing raw text into tensor

        Args:
            `text` (`str`): raw string of sentence

        Returns:
            `output` (`torch.Tensor`): tensor representation of sentence with shape `[sequence_length]`
        """
        tokenized_text = self.tokenizer(text)
        # <SOS> ... <EOS>
        output = [self.word2idx['<SOS>']] + [
            self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>']
            for token in tokenized_text
        ] + [self.word2idx['<EOS>']]
        return torch.tensor(output)
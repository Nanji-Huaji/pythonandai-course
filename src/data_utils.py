import torch
import re
from collections import Counter
import os

# --- Tokenizer ---
def basic_english_tokenizer(text):
    # Simple tokenizer: lowercase and split by words (basic regex)
    return re.findall(r'\b\w+\b', text.lower())

# --- Vocab Class ---
class Vocab:
    def __init__(self, token_to_idx, specials):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        self.unk_index = token_to_idx.get("<unk>")
        self.pad_index = token_to_idx.get("<pad>")
        self.specials = specials
        
    def __getitem__(self, token):
        # Return index if exists, else unk_index
        return self.token_to_idx.get(token, self.unk_index)
        
    def __len__(self):
        return len(self.token_to_idx)
        
    def lookup_token(self, idx):
        return self.idx_to_token.get(idx, "<unk>")

def build_vocab_from_iterator_custom(iterator, specials=["<unk>", "<pad>"]):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    
    # Sort by freq
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    token_to_idx = {}
    idx = 0
    # Add specials first
    for s in specials:
        token_to_idx[s] = idx
        idx += 1
        
    # Add rest (Top 30000 words)
    max_vocab = 30000 
    for token, freq in sorted_by_freq:
        if idx >= max_vocab:
            break
        if token not in token_to_idx:
            token_to_idx[token] = idx
            idx += 1
            
    return Vocab(token_to_idx, specials)

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def save_vocab(vocab, path):
    torch.save(vocab, path)

def load_vocab(path):
    # weights_only=False is needed because we are loading a custom class (Vocab)
    # Since we created this file locally, it is safe.
    return torch.load(path, weights_only=False)

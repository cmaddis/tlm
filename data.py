import numpy as np
import requests
try:
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/files/TinyStories-train-subset.txt")
    response.raise_for_status() 
    TRAIN = response.text
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/files/TinyStories-valid-subset.txt")
    response.raise_for_status() 
    VAL = response.text
except requests.RequestException as e:
    print(f"Error reading file from URL: {e}")

# create a mapping from characters to integers
ALPHABET = set(TRAIN+VAL)
ALPHABET_LIST = sorted(list(ALPHABET))
ctoi = { ch:i for i,ch in enumerate(ALPHABET_LIST) }
itoc = { i:ch for i,ch in enumerate(ALPHABET_LIST) }
encode = lambda s: [ctoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itoc[i] for i in l]) # decoder: take a list of integers, output a string

# encode the data
TRAIN_TOKENS = np.array(encode(TRAIN), dtype=np.int32)
VAL_TOKENS = np.array(encode(VAL), dtype=np.int32)

# A batching function that returns input, output arrays of size (batch_size, context_len)
def get_batch(batch_size, context_len, split):
    # generate a small batch of data of inputs x and targets y
    data = TRAIN_TOKENS if split == "train" else VAL_TOKENS
    ix = np.random.randint(0, len(data) - context_len, (batch_size,))
    X = np.stack([data[i:i+context_len] for i in ix])
    Y = np.stack([data[i+1:i+context_len+1] for i in ix])
    return X, Y
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
charset = set(TRAIN+VAL)
charlist = sorted(list(charset))
ctoi = { ch:i for i,ch in enumerate(charlist) }
itoc = { i:ch for i,ch in enumerate(charlist) }
encode = lambda s: [ctoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itoc[i] for i in l]) # decoder: take a list of integers, output a string

# encode the data
TRAIN_ARR = np.array(encode(TRAIN), dtype=np.int32)
VAL_ARR = np.array(encode(VAL), dtype=np.int32)

# A batching function that returns input, output arrays of size (batch_size, block_size)
def get_batch(batch_size, block_size, split):
    # generate a small batch of data of inputs x and targets y
    data = TRAIN_ARR if split == "train" else VAL_ARR
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
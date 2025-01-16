# A Tiny Large Model

A minimal implementation of a transformer-based language model, following [nanoGPT](https://github.com/karpathy/nanoGPT) very closely. This implementation emphasizes pedagogy, using [JAX](https://github.com/jax-ml/jax) and [Penzai's](https://github.com/google-deepmind/penzai) named array system to make the attention mechanism more intuitive and easier to understand. This project was developed for [CSC2541: Large Models](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/) at the University of Toronto.

## Key Features

- Self-contained implementation of a transformer model in JAX
- Uses named arrays to make tensor operations more semantically meaningful
- Trains on a subset of the [TinyStories dataset](https://arxiv.org/abs/2305.07759) for lightweight experimentation

## Understanding the Code
The cleanest part of the code base is `model.py`, which contains the implementation of the Transformer model. We recommend you focus your efforts there.

We use Penzai's NamedArray to make Transformer's computations more clear. Although they require a bit of getting used to, named axes allow you to focus on the key semantics of the computation, while ignoring the structure of the data. The best place to start learning about Penzai's named axes [is here](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html). 

For example, in this pseudo-code for a (non-causal) attention mechanism, instead of keeping track of tensor axes, we explicitly name our dimensions. This is what it looks like:

    # Every key interacts with every query via the dot of their embeddings
    attn_logits = nmap(jnp.dot)(q.untag("embed"), k.untag("embed"))

    # Scale the attention logits to ensure constant variance
    feature_dim = q.named_shape["embed"]
    attn_logits = attn_logits / np.sqrt(feature_dim)

    # Compute a distribution over keys for each query
    attn_dist = nmap(jax.nn.softmax)(attn_logits.untag("kseq))
    attn_dist = attn_dist.tag("kseq")

    # The value returned for each query is the average value
    # indexed by the keys, under the attention distribution
    out = nmap(jnp.dot)(attn_dist.untag("kseq"), v.untag("kseq"))


This makes it a bit more clear which axes are participating in which computation.

## Getting Started

1. Install dependencies: `jax`, `penzai`, `optax`, `tqdm`, `requests`, `numpy`
2. Run the training script: `python train.py`

The model will train on a subset of TinyStories and generate sample text.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Mistakes

Please let us know if you find any mistakes! There probably are some.
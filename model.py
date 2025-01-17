'''
A Transformer model implemented in JAX with Penzai named array system.

This code contains assertion statements throughout, to help students keep track of the expected shape of arrays.
'''
import numpy as np
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.core.named_axes import wrap, nmap
from convenience import (sample_along, axis_names_are, axis_names_contain)

def generate_from_transformer(key, params, inp, context_size, max_new_tokens):
    """Autoregressively generates new tokens by sampling from the model's predictions, using a sliding context window if needed.
    
    WARNING: This implementation emphasizes pedagogy, and is not very efficient."""
    assert axis_names_are(inp, {"batch", "seq"})

    generated = inp
    for _ in range(max_new_tokens):
        # get context (at most context_size)
        context_window = (generated if generated.named_shape["seq"] <= context_size else 
                        generated[{"seq" : pz.slice[-context_size:]}])
        
        # Get predictions (my imeplementation is very wasteful)
        log_preds = get_log_predictions(params, context_window)

        # Take the last time step prediction
        log_pred = log_preds[{"seq" : pz.slice[-1:]}]

        # Sample categoricals along the alphabet dimension
        key, subkey = jax.random.split(key)
        next_tokens = jax.jit(sample_along, static_argnums=2)(subkey, log_pred, "alphabet")

        # concatenate along the seq dimension to the running sequence
        generated = pz.nx.concatenate((generated, next_tokens), "seq")
    return generated

@jax.jit
def get_log_predictions(params, inp):
    """Computes log-probabilities over the alphabetulary for each position in the input sequence using the full Transformer model."""
    assert axis_names_are(inp, {"batch", "seq"})

    seq_len = inp.named_shape["seq"]

    # Token and position embeddings
    tok_emb = params['token_embedding'][{"alphabet" : inp}] # embed tokens
    assert axis_names_are(tok_emb, {"batch", "seq", "embed"})
    pos_emb = params['position_embedding'][{"seq" : pz.nx.arange("seq", seq_len)}]
    assert axis_names_are(pos_emb, {"seq", "embed"})
    emb = tok_emb + pos_emb
    assert axis_names_are(emb, {"batch", "seq", "embed"})

    # Apply transformer blocks
    for b, block_params in enumerate(params['blocks']):
        emb = transformer_block(block_params, emb)

    # Final layer norm
    emb = renorm_along(params['ln_f'], emb, "embed")

    # Project into alphabet
    logits = affine_along(params["lm_head"], emb, "embed")

    # log softmax over "embed", which is length "alphabet"
    log_preds = nmap(jax.nn.log_softmax)(logits.untag("embed"))
    log_preds = log_preds.tag("alphabet")
    return log_preds

def transformer_block(params, emb):
    """Processes input through a complete Transformer block: layer norm → self-attention → residual → layer norm → feed-forward → residual."""
    assert axis_names_are(emb, {"batch", "seq", "embed"})

    # LAYER NORM
    # Renormalizing along the "embed" dimension is layer norm
    ln1_out = renorm_along(params['ln1'], emb, "embed")

    # MULTIHEADED CAUSAL SELF ATTENTION
    attn_out = multiheaded_selfattention(params['attn'], ln1_out, causal=True)

    # RESIDUAL CONNECTION
    emb = emb + attn_out

    # LAYER NORM
    # Renormalizing along the embed dimension is layer norm
    ln2_out = renorm_along(params['ln2'], emb, "embed")

    # FEED FORWARD MLP along the embed dimension
    ff_out = mlp_along(params['ffwd'], ln2_out, "embed")
    emb = emb + ff_out

    return emb

def multiheaded_selfattention(params, emb, causal=True):
    assert axis_names_are(emb, {"batch", "seq", "embed"})

    # Apply three linear maps to the embeddings independently across "seq"
    # These are now our keys and queries and values for self-attention!
    queries = nmap(jnp.dot)(params["Wq"].untag("embed"), emb.untag("embed"))
    keys = nmap(jnp.dot)(params["Wk"].untag("embed"), emb.untag("embed"))
    values = nmap(jnp.dot)(params["Wv"].untag("embed"), emb.untag("embed"))

    # As this is SELF-attention, the queries and keys both have the same dimension "seq". 
    # But, we want (key, query) pairs to interact, even if the key and query index
    # is different. To accomplish this with the named array system, we need to rename 
    # the "seq" axis for both.
    queries = queries.untag("seq").tag("query")
    keys = keys.untag("seq").tag("key")
    values = values.untag("seq").tag("key")

    assert axis_names_are(queries, {"batch", "query", "head", "embed/head"})
    assert axis_names_are(keys, {"batch", "key", "head", "embed/head"})
    assert axis_names_are(values, {"batch", "key", "head", "embed/head"})

    # Compute scores by taking the inner product of every key with every query
    # This checks "alignment".
    scores = nmap(jnp.dot)(queries.untag("embed/head"), keys.untag("embed/head"))

    # Scale the attention logits to avoid saturating the softmax
    feature_dim = queries.named_shape["embed/head"]
    scores = scores / np.sqrt(feature_dim)

    if causal:
        # For causal attention, we want to mask out (set to -inf) any (key, query) pair
        # such that key's index is greater than query's index, i.e., key is in the future.
        # This ensures the softmax will give zero probability to attending to the future.

        # Get the sequence indices for the queries and keys to enable causal masking
        query_dim = queries.named_shape["query"]
        key_dim = keys.named_shape["key"]
        k_idxs = pz.nx.arange("key", key_dim)
        q_idxs = pz.nx.arange("query", query_dim)

        # Set the scores to -inf where q < k
        scores = nmap(jnp.where)(
        q_idxs < k_idxs,  # Broadcasts across other dimensions
        float('-inf'),
        scores,
        )

    # Compute a probability distribution over keys for each query
    # This is the "soft" index, aka attention!
    attn_dist = nmap(jax.nn.softmax)(scores.untag("key"))
    attn_dist = attn_dist.tag("key")

    # Taking the inner product along the key axis with the attention distribution
    # returns the average value, with each key contributing in proportion to 
    # its attention weight
    # the output will have named axes {"query", "embed"} and is a floating-point array
    emb = nmap(jnp.dot)(attn_dist.untag("key"), values.untag("key"))

    # We're going to rearrange the dimensions now to bring us back to convention.
    # First, we rename "query", which was a holdover from our attention convention
    emb = emb.untag("query").tag("seq")

    # emb named axes should now include {"seq", "head", "embed/head"}
    # We can concatenate "head" and "embed/head" axes by reshaping:
    emb = emb.untag("head", "embed/head").flatten().tag("embed")

    # emb named axes should now include {"seq", "embed"}

    # Final affine layer
    emb = affine_along(params["aff"], emb, "embed")
    return emb

def renorm_along(params, x, axis):
    """Renormalizes along the specified axis with learnable scale and bias parameters."""
    assert axis_names_contain(x, {axis})

    # Convert to positional axis for computation
    # You can now think of x as if it's a single axis-dimensional array
    x = x.untag(axis)
    scale = params["scale"].untag(axis)
    bias = params["bias"].untag(axis)
    epsilon = 1e-5

    # Normalize along axis
    mean = x.mean()
    std = nmap(jnp.sqrt)(x.var() + epsilon)
    out = (x - mean) / std

    # Apply learnable scale and bias
    return (scale * out + bias).tag(axis)

def mlp_along(params, x, feature_axis):
    """Applies a two-layer feed-forward network with ReLU activation along the feature axis."""
    assert axis_names_contain(x, {feature_axis})

    x = affine_along(params["fc1"], x, feature_axis)
    x = nmap(jax.nn.relu)(x)
    x = affine_along(params["fc2"], x, feature_axis)
    return x

def affine_along(params, x, feature_axis):
    """Applies an affine transformation (Wx + b) along the specified feature axis."""
    assert axis_names_contain(x, {feature_axis})
    # Unpack the parameters
    W = params["W"]
    b = params["b"]

    # We use the convention that linear layers applied
    # to feature_axis have named axes 
    # {feature_axis+ "_in", feature_axis+ "_out"} and
    assert axis_names_are(W, {feature_axis+"_in", feature_axis+"_out"})
    assert axis_names_are(b, {feature_axis+"_out"})

    # Apply affine transformation
    out = nmap(jnp.dot)(
        W.untag(feature_axis+ "_in"),
        x.untag(feature_axis),
    )
    out = out + b

    # Use the convention that we do not rename the feature_axis
    # of the embeddings, so we have to rename the axis
    out = out.untag(feature_axis+"_out").tag(feature_axis)
    return out

def param_init(key, config):
    """Initializes a Transformer model's parameters using standard initialization schemes from PyTorch."""
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_dim = config["embed_dim"]
    context_len = config["context_len"]
    alphabet_size = config["alphabet_size"]
    head_size = embed_dim // n_heads
    hidden_dim = 4 * embed_dim

    # Make more RNGs
    keys = jax.random.split(key, n_layers+4)

    # PyTorch default initialization for embeddings is N(0, 1)
    token_embedding = wrap(jax.random.normal(keys[0], (alphabet_size, embed_dim))).tag("alphabet", "embed")
    position_embedding = wrap(jax.random.normal(keys[1], (context_len, embed_dim))).tag("seq", "embed")

    # PyTorch default initialization for linear layers is U(-sqrt(k), sqrt(k))
    # where k = 1/in_features
    def sample_uniform(key, in_features, shape):
        k = 1/in_features
        u = 2 * jax.random.uniform(key, shape) - 1
        return jnp.sqrt(k) * u
    
    # Layer parameters
    params = {
        'token_embedding': token_embedding,
        'position_embedding': position_embedding
    }
    params['ln_f'] = {
            'scale': wrap(jnp.ones(embed_dim)).tag("embed"),
            'bias': wrap(jnp.zeros(embed_dim)).tag("embed")
    }

    shape = (embed_dim, alphabet_size)
    params['lm_head'] = {
        'W': wrap(sample_uniform(keys[2], shape[0], shape)).tag("embed_in", "embed_out"),
        'b': wrap(sample_uniform(keys[3], shape[0], shape[1])).tag("embed_out")
    }
    params['blocks'] = []

    # Initialize transformer blocks
    for i in range(n_layers):
        block_keys = jax.random.split(keys[i + 4], 9)

        block = {}
        block['ln1'] = {
            'scale': wrap(jnp.ones(embed_dim)).tag("embed"),
            'bias': wrap(jnp.zeros(embed_dim)).tag("embed")
        }

        shape = (embed_dim, head_size, n_heads)
        block['attn'] = {
            'Wq': wrap(sample_uniform(block_keys[0], shape[0], shape)).tag("embed", "embed/head", "head"),
            'Wk': wrap(sample_uniform(block_keys[1], shape[0], shape)).tag("embed", "embed/head", "head"),
            'Wv': wrap(sample_uniform(block_keys[2], shape[0], shape)).tag("embed", "embed/head", "head"),
        }
        shape = (embed_dim, embed_dim)
        block['attn']['aff'] = {
            'W': wrap(sample_uniform(block_keys[3], shape[0], shape)).tag("embed_in", "embed_out"),
            'b': wrap(sample_uniform(block_keys[4], shape[0], shape[1])).tag("embed_out")
        }

        block['ln2'] = {
            'scale': wrap(jnp.ones(embed_dim)).tag("embed"),
            'bias': wrap(jnp.zeros(embed_dim)).tag("embed")
        }

        block['ffwd'] = {}
        shape = (embed_dim, hidden_dim)
        block['ffwd']['fc1'] = {
            'W': wrap(sample_uniform(block_keys[5], shape[0], shape)).tag("embed_in", "embed_out"),
            'b': wrap(sample_uniform(block_keys[6], shape[0], shape[1])).tag("embed_out")
        }
        shape = (hidden_dim, embed_dim)
        block['ffwd']['fc2'] = {
            'W': wrap(sample_uniform(block_keys[7], shape[0], shape)).tag("embed_in", "embed_out"),
            'b': wrap(sample_uniform(block_keys[8], shape[0], shape[1])).tag("embed_out")
        }

        params['blocks'].append(block)

    return params
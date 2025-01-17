from penzai import pz
from penzai.core.named_axes import wrap, nmap
import jax
import jax.numpy as jnp

def cross_entropy_along(targets, logits, axis):
  """Cross entropy of the predictions"""
  assert axis_names_are(logits, set(targets.named_shape.keys()).union({"alphabet"}))

  # Take the log_predictions along the alphabet axis for each target_token index
  log_losses = logits[{axis : targets}]

  # Unwrap named axis functionality and reduce down a single number
  loss = -jnp.mean(log_losses.unwrap(*log_losses.named_shape.keys()))
  return loss

def sample_along(key, logits, axis):
    """Sample from a categorical distribution along axis."""
    assert axis_names_contain(logits, {axis})

    subkeys = pz.nx.random_split(wrap(key),
        [(name, dim) for (name, dim) in logits.named_shape.items()
            if name != axis]
    )
    return nmap(jax.random.categorical)(subkeys, logits.untag(axis))

def is_fully_named(x):
    return x.positional_shape == ()

def axis_names_are(x, names):
    return names == set(x.named_shape.keys())

def axis_names_contain(x, names):
    return names <= set(x.named_shape.keys())

def get_innout_axes(namedarray):
    ax1, ax2 = tuple(namedarray.named_shape.keys())
    if ax1.endswith("_in") and ax2.endswith("_out"):
        return ax1, ax2
    elif ax2.endswith("_in") and ax1.endswith("_out"):
        return ax2, ax1
    else:
        raise ValueError(f"Named axes must end with '_in' and '_out', but got '{ax1}' and '{ax2}'. ")
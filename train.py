import numpy as np
import jax
from penzai.core.named_axes import wrap
import optax
import tqdm
from model import param_init, get_log_predictions, generate_from_transformer
from convenience import axis_names_are, cross_entropy_along
import data

# Initialize parameters and optimizer
config = {
    "batch_size" : 16,
    "context_len" : 32,
    "embed_dim" : 64,
    "n_heads" : 4,
    "n_layers" : 4,
    "learning_rate" : 1e-3,
    "alphabet_size" : len(data.ALPHABET),
    "max_iters" : 5000,
    "eval_interval" : 500,
    "eval_iters" : 200,
}

key = jax.random.key(0)
key, subkey = jax.random.split(key)
params = param_init(subkey, config)

# Create optimizer
optimizer = optax.adam(config["learning_rate"])
opt_state = optimizer.init(params)

# Define training step
@jax.jit
def empirical_risk(params, input_tokens, target_tokens):
    log_preds = get_log_predictions(params, input_tokens)
    assert axis_names_are(log_preds, {"batch", "seq", "alphabet"})
    return cross_entropy_along(target_tokens, log_preds, "alphabet")

@jax.jit
def train_step(params, input_tokens, target_tokens, opt_state):
    assert axis_names_are(input_tokens, {"batch", "seq"})
    assert axis_names_are(target_tokens, {"batch", "seq"})
    loss, grads = jax.value_and_grad(empirical_risk)(params, input_tokens, target_tokens)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# Logging
train_losses = []
val_losses = []

# Training loop
pbar = tqdm.tqdm(range(config["max_iters"]), desc="Training...                ")
for iter in pbar:
    # Get a batch of data
    input_tokens, target_tokens = data.get_batch(config["batch_size"], config["context_len"], "train")

    # Wrap the tokens into named arrays
    input_tokens = wrap(input_tokens).tag("batch", "seq")
    target_tokens = wrap(target_tokens).tag("batch", "seq")

    # Train step
    params, opt_state, loss = train_step(params, input_tokens, target_tokens, opt_state)

    # Logging
    train_losses.append(float(loss))

    # Evaluate on validation set
    if iter % config["eval_interval"] == 0:
        losses = []
        for _ in range(config["eval_iters"]):
            # Get a batch of data
            input_tokens, target_tokens = data.get_batch(config["batch_size"], config["context_len"], "val")

            # Wrap the tokens into named arrays
            input_tokens = wrap(input_tokens).tag("batch", "seq")
            target_tokens = wrap(target_tokens).tag("batch", "seq")

            loss = empirical_risk(params, input_tokens, target_tokens)
            losses.append(float(loss))

        val_loss = sum(losses) / len(losses)
        val_losses.append(val_loss)
    pbar.set_description(f"train:{train_losses[-1]:.4f}, val:{val_losses[-1]:.4f}")

n_tokens = 2000
prompt_tokens = np.array([[0]], dtype=int)
prompt_tokens = wrap(prompt_tokens).tag("batch", "seq")
generated_tokens = generate_from_transformer(key, params, prompt_tokens, config["context_len"], n_tokens)
completion = data.decode(generated_tokens[{"batch" : 0}].untag("seq").unwrap().tolist())
print(f"Sample:\n{completion}")

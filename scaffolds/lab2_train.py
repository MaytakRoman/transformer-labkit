"""Student assignment 2. Implement and train a transformer on generated data."""

def build_model(config):
    # TODO: embedding, position representation, transformer, vocabulary logits.
    raise NotImplementedError('Implement model')

def train(config, train_data, valid_data, run_directory):
    # TODO: CPU seed streams, masked loss, AdamW, validation, checkpoint.
    # Never select hyperparameters by test results. Do not feed latent to model.
    raise NotImplementedError('Implement training')

def evaluate(model, split_data):
    # TODO: token accuracy and loss over loss_mask only, predictions with IDs.
    raise NotImplementedError('Implement evaluation')

# Save config, optimizer/scheduler/RNG state, predictions and paired activations.
# Check overfit on a tiny set and equivalence of uninterrupted/resumed updates.

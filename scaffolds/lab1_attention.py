"""Student assignment 1. Use attention_inputs.npz and its config temperature."""

def masked_softmax(logits, allowed, axis=-1):
    # TODO: stable softmax on allowed cells; reject a fully masked row.
    raise NotImplementedError('Implement masked_softmax')

def multihead_attention(q, k, v, allowed, temperature):
    # TODO: shapes [heads, tokens, head_dim]; scale by sqrt(head_dim)*temperature.
    # Return outputs [heads, tokens, head_dim] and weights [heads, tokens, tokens].
    raise NotImplementedError('Implement multihead_attention')

# Required evidence: manual small case; zero masked mass; row sums;
# future-token intervention for causal masks; incorrect mask negative control.

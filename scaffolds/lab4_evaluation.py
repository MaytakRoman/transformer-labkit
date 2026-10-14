"""Student assignment 4. Inputs explicitly contain simulated predictions/resources."""

def select_threshold(valid_predictions, valid_targets):
    # TODO: freeze threshold before evaluating test.
    raise NotImplementedError('Implement validation-only threshold selection')

def metrics(probabilities, targets, threshold):
    # TODO: confusion matrix, accuracy/F1/BACC/MCC, Brier and specified ECE bins.
    raise NotImplementedError('Implement metrics with degenerate-case policy')

def admit(config_metrics, constraints):
    # TODO: require every mandatory constraint, then rank admissible configs.
    raise NotImplementedError('Implement admission rule')

# Report controls for stress, ablation and destructive predictions.
# These score controls do not measure causal effects of a real neural component.
# For resource benchmarking replace simulated values with timed device trials.

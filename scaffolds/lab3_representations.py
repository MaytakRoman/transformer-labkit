"""Student assignment 3. First validate on synthetic controls, then lab 2 states."""

def linear_cka(x, y):
    # TODO: assert paired IDs in caller; center, form Gram matrices, normalize.
    raise NotImplementedError('Implement linear CKA')

def orthogonal_alignment(x, y):
    # TODO: center, SVD cross-covariance, orthogonal map and residual.
    raise NotImplementedError('Implement Procrustes alignment')

def pca_train_only(train, valid, test, components):
    # TODO: fit mean and basis on train only; apply unchanged to valid/test.
    raise NotImplementedError('Implement PCA')

# Compare paired/unpaired objects; report explained variance and controls.
# Synthetic control representations are not outputs of a trained transformer.

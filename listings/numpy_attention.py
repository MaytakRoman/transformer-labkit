"""Readable reconstruction of printed fragments 7.1 and 7.2.
The DOCX merged line breaks. This file restores layout and adds smoke checks.
"""
import numpy as np

def masked_softmax(logits, allowed, axis=-1):
    if not np.all(allowed.any(axis=axis)):
        raise ValueError('fully masked row')
    x = np.where(allowed, logits, -np.inf)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.where(allowed, np.exp(x - m), 0.0)
    return e / e.sum(axis=axis, keepdims=True)

def attention(q, k, v, allowed):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    weights = masked_softmax(scores, allowed)
    return weights @ v, weights

def checks():
    r=np.random.default_rng(42)
    q,k,v=[r.normal(size=(5,4)) for _ in range(3)]
    mask=np.tri(5,dtype=bool);out,w=attention(q,k,v,mask)
    np.testing.assert_allclose(w.sum(-1),1)
    assert np.all(w[~mask]==0)
    changed=k.copy();changed[3:]+=100
    np.testing.assert_allclose(out[:3],attention(q,changed,v,mask)[0][:3])
    try:masked_softmax(np.zeros((2,2)),np.zeros((2,2),bool))
    except ValueError:pass
    else:raise AssertionError('Fully masked row accepted')
    print('NumPy fragments 7.1 / 7.2: checks passed')

if __name__=='__main__':checks()

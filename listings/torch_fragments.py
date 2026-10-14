"""Restored, parameterized fragments 7.3--7.6; requires optional PyTorch.

These are API examples, not a completed model or the lab 2 training solution.
Checkpoint helper extends the printed example to include additional RNG streams.
"""
import os
import random
import numpy as np
import torch

def check_batch(tokens, mask):
    assert tokens.ndim == 2 and mask.shape == tokens.shape
    assert tokens.dtype == torch.long
    assert mask.dtype == torch.bool
    assert torch.all(mask.any(dim=1))

def training_step(model, optimizer, scheduler, criterion,
                  tokens, attention_mask, targets, valid, max_norm):
    if not valid.any():raise ValueError('no supervised positions')
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens, attention_mask)
    loss = criterion(logits[valid], targets[valid])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())

def check_gradient(parameter,index,loss_fn,analytic,eps=1e-6,atol=1e-7,rtol=1e-4):
    with torch.no_grad():
        old=parameter[index].item()
        try:
            parameter[index]=old+eps;lp=loss_fn().item()
            parameter[index]=old-eps;lm=loss_fn().item()
        finally:parameter[index]=old
    numeric=(lp-lm)/(2*eps)
    assert abs(numeric-analytic)<=atol+rtol*abs(numeric)
    return numeric

def save_checkpoint(model,optimizer,scheduler,step,effective_config,
                    temporary_path,checkpoint_path,sampler_state):
    # Supply application-owned Generator and worker states in sampler_state.
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'scheduler':scheduler.state_dict(),'step':step,
           'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
           'python_rng':random.getstate(),'config':effective_config,
           'sampler_state':sampler_state,
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state,temporary_path)
    os.replace(temporary_path,checkpoint_path)

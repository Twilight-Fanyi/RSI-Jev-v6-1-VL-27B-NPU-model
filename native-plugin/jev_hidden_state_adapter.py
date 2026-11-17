"""Portable hidden-state -> RSI-Jev typed decision adapter.

This module intentionally does not pretend to be the vLLM HTTP server. It defines the
missing contract between vLLM's final normalized token states and the verified
OptionScorer heads: decision_index, option_span_start/end, option_mask and mode_id.
It can be imported by a custom vLLM model/runner once the engine exposes final states.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from jev_scorer import OptionXAttn

@dataclass
class HiddenStateBatch:
    states: torch.Tensor                 # (B,T,H), final normalized hidden states
    decision_index: torch.Tensor         # (B,)
    option_span_start: torch.Tensor      # (B,K)
    option_span_end: torch.Tensor        # (B,K)
    option_mask: torch.Tensor            # (B,K), bool
    mode_id: Optional[torch.Tensor] = None

def pool_spans(batch: HiddenStateBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool each [start,end) option span and gather the decision state."""
    h=batch.states.float(); B,T,H=h.shape; K=batch.option_mask.shape[1]
    b=torch.arange(B,device=h.device)
    decision=h[b,batch.decision_index]
    # Build a small dense span mask; K is <=255 in the competition protocol.
    t=torch.arange(T,device=h.device)[None,None,:]
    inside=((t>=batch.option_span_start[...,None]) & (t<batch.option_span_end[...,None]) & batch.option_mask[...,None])
    denom=inside.sum(-1,keepdim=True).clamp_min(1).float()
    options=torch.einsum('bkt,bth->bkh',inside.float(),h)/denom
    return decision,options

def score(batch: HiddenStateBatch, head: OptionXAttn, log_temperature: float=0.0) -> torch.Tensor:
    dh,oh=pool_spans(batch)
    z=head(dh,oh,batch.option_mask)
    if log_temperature:
        z=z/torch.exp(torch.as_tensor(log_temperature,device=z.device,dtype=z.dtype))
    return z

def probabilities(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p=torch.softmax(z.float().masked_fill(~mask,float('-inf')),dim=-1)
    return p.masked_fill(~mask,0.0)

class FinalStateCapture:
    """Forward-hook capture for a vLLM/Transformers final norm module."""
    def __init__(self): self.value=None
    def hook(self, module, inputs, output):
        x=output[0] if isinstance(output,(tuple,list)) else output
        self.value=x

@contextmanager
def capture_final_norm(model: nn.Module):
    """Attach to the first module whose name ends in `.norm` under language_model.

    A custom vLLM runner can use this around its model call; the engine still needs
    to pass the captured tensor to score(). No model weights or engine state change.
    """
    cap=FinalStateCapture(); target=None
    for name,m in model.named_modules():
        if name.endswith('language_model.norm') or name.endswith('model.norm'):
            target=m; break
    if target is None: raise LookupError('final language-model norm not found')
    handle=target.register_forward_hook(cap.hook)
    try: yield cap
    finally: handle.remove()


def encode_option_prompt(
    tokenizer,
    state: str,
    instructions: str,
    options: list[str],
    criteria: dict[str, object] | None = None,
    answer_cue: str = "Answer:",
    option_pool_own_tokens: bool = False,
) -> dict[str, object]:
    """Encode one typed decision prompt and return spans for HiddenStateBatch."""
    criteria = criteria or {}
    head = f"{state}\n\n{instructions}\nOptions:\n" if state.strip() else f"{instructions}\nOptions:\n"
    input_ids = list(tokenizer(head, add_special_tokens=False)["input_ids"])
    starts, ends = [], []
    for key in options:
        criterion = criteria.get(key)
        block = f"- {key}: {criterion}" if criterion else f"- {key}"
        start = len(input_ids)
        input_ids.extend(tokenizer("\n" + block, add_special_tokens=False)["input_ids"])
        starts.append(start)
        ends.append(len(input_ids))
    input_ids.extend(tokenizer("\n\n" + answer_cue, add_special_tokens=False)["input_ids"])
    decision_index = len(input_ids) - 1
    if option_pool_own_tokens:
        for idx, key in enumerate(options):
            own_prefix = len(tokenizer("\n- " + key + ":", add_special_tokens=False)["input_ids"])
            starts[idx] = min(starts[idx] + own_prefix, ends[idx] - 1)
    return {
        "input_ids": input_ids,
        "decision_index": decision_index,
        "option_span_start": starts,
        "option_span_end": ends,
        "option_mask": [True] * len(options),
        "options": list(options),
    }

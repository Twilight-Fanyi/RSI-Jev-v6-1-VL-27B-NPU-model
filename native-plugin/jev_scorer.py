"""Native PyTorch typed RSI-Jev v6.1 OptionScorer heads.
Weights match source-rsi-jev/rsijev/mlx/heads.py option_xattn,sum,4 heads.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file

class OptionXAttn(nn.Module):
    def __init__(self, hidden=5120, heads=4):
        super().__init__(); self.q=nn.Linear(hidden,hidden); self.k=nn.Linear(hidden,hidden); self.v=nn.Linear(hidden,hidden)
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True); self.proj=nn.Linear(hidden,1)
    @classmethod
    def from_state(cls, sd, prefix=''):
        m=cls(); d={k[len(prefix):] if prefix and k.startswith(prefix) else k:v for k,v in sd.items()}
        with torch.no_grad():
            for n in ('q','k','v'):
                m.__getattr__(n).weight.copy_(d[n+'.weight']); m.__getattr__(n).bias.copy_(d[n+'.bias'])
            m.attn.in_proj_weight.copy_(d['attn.in_proj_weight']); m.attn.in_proj_bias.copy_(d['attn.in_proj_bias'])
            m.attn.out_proj.weight.copy_(d['attn.out_proj.weight']); m.attn.out_proj.bias.copy_(d['attn.out_proj.bias'])
            m.proj.weight.copy_(d['proj.weight']); m.proj.bias.copy_(d['proj.bias'])
        return m.eval()
    def forward(self, decision_h, option_h, option_mask):
        q=self.q(decision_h).unsqueeze(1); k=self.k(option_h); v=self.v(option_h)
        ctx,_=self.attn(q,k,v,key_padding_mask=~option_mask,need_weights=False)
        return self.proj(v+ctx).squeeze(-1).masked_fill(~option_mask,float('-inf'))

def load_bundle(root):
    root=Path(root); main=OptionXAttn.from_state(load_file(str(root/'scorer.safetensors')))
    aux=load_file(str(root/'aux_scorers.safetensors')); auxm={}
    for L in (48,56): auxm[L]=OptionXAttn.from_state(aux,prefix=f'{L}.')
    cal=load_file(str(root/'calibration.safetensors'))
    return main,auxm,cal

def smoke(root):
    main,aux,cal=load_bundle(root); torch.manual_seed(7); dh=torch.randn(2,5120); oh=torch.randn(2,5,5120); mask=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]],dtype=torch.bool)
    out={'shape':list(main(dh,oh,mask).shape),'main':{},'aux':{}}
    for tag,m in [('main',main),('48',aux[48]),('56',aux[56])]:
        z=m(dh,oh,mask); p=torch.softmax(z,dim=-1); out[tag]={'finite':bool(torch.isfinite(z[mask]).all()),'masked_neg_inf':bool(torch.isneginf(z[~mask]).all()),'prob_sum':p.masked_fill(~mask,0).sum(-1).tolist()}
    out['calibration_keys']=sorted(cal.keys()); return out
if __name__=='__main__':
    import sys; print(json.dumps(smoke(sys.argv[1]),sort_keys=True))

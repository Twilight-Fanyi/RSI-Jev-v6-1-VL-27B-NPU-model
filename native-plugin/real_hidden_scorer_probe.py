import json, sys, torch
from pathlib import Path
sys.path.insert(0, "/workspace/jev-adaptation/model-44/native-plugin")
from jev_scorer import load_bundle
from jev_hidden_state_adapter import HiddenStateBatch, score, probabilities

root=Path("/workspace/jev-adaptation/model-44/weights")
states=torch.load("/workspace/jev-adaptation/model-44/evidence/C-hidden-real.pt", map_location="cpu", weights_only=True)
if states.ndim == 2:
    states=states.unsqueeze(0)
starts=torch.tensor([[0,1]], dtype=torch.long)
ends=torch.tensor([[1,3]], dtype=torch.long)
mask=torch.tensor([[True,True]])
dec=torch.tensor([2], dtype=torch.long)
batch=HiddenStateBatch(states, dec, starts, ends, mask)
main, aux, cal=load_bundle(root)
z=score(batch, main)
za=score(batch, aux[48])
with open(root/"calibration.json", encoding="utf-8") as f:
    calibration=json.load(f)
main_logt=float(calibration["main_logT"])
aux48_logt=float(calibration["exits"]["48"]["logT"])
p=probabilities(z, mask)
pa=probabilities(za, mask)
zc=z/torch.exp(torch.tensor(main_logt))
zac=za/torch.exp(torch.tensor(aux48_logt))
pc=probabilities(zc, mask)
pac=probabilities(zac, mask)
print(json.dumps({
  "states_shape": list(states.shape),
  "decision_index": dec.tolist(),
  "spans": [[0,1],[1,3]],
  "main_logits": z.tolist(),
  "main_probabilities": p.tolist(),
  "main_logT": main_logt,
  "main_calibrated_logits": zc.tolist(),
  "main_calibrated_probabilities": pc.tolist(),
  "aux48_logits": za.tolist(),
  "aux48_probabilities": pa.tolist(),
  "aux48_logT": aux48_logt,
  "aux48_calibrated_logits": zac.tolist(),
  "aux48_calibrated_probabilities": pac.tolist(),
  "finite": bool(torch.isfinite(z).all() and torch.isfinite(za).all()),
  "prob_sum": [float(p.sum()),float(pa.sum()),float(pc.sum()),float(pac.sum())],
}, ensure_ascii=False, sort_keys=True))

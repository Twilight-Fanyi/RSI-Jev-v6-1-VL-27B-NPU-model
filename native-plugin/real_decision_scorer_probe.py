import json, sys, torch
from pathlib import Path
sys.path.insert(0, "/workspace/jev-adaptation/model-44/native-plugin")
from jev_scorer import load_bundle
from jev_hidden_state_adapter import HiddenStateBatch, score, probabilities
root=Path("/workspace/jev-adaptation/model-44/weights")
enc=json.load(open("/workspace/jev-adaptation/model-44/evidence/C-real-decision-encoding.json"))
states=torch.load("/workspace/jev-adaptation/model-44/evidence/C-real-decision-hidden.pt",map_location="cpu",weights_only=True)
if states.ndim==2: states=states.unsqueeze(0)
starts=torch.tensor([enc["option_span_start"]],dtype=torch.long)
ends=torch.tensor([enc["option_span_end"]],dtype=torch.long)
mask=torch.tensor([enc["option_mask"]],dtype=torch.bool)
dec=torch.tensor([enc["decision_index"]],dtype=torch.long)
batch=HiddenStateBatch(states,dec,starts,ends,mask)
main,aux,cal=load_bundle(root)
z=score(batch,main); z48=score(batch,aux[48]); z56=score(batch,aux[56])
calib=json.load(open(root/"calibration.json"))
def cal(z,logt): return probabilities(z/torch.exp(torch.tensor(float(logt))),mask)
p=cal(z,calib["main_logT"]); p48=cal(z48,calib["exits"]["48"]["logT"]); p56=cal(z56,calib["exits"]["56"]["logT"])
print(json.dumps({
 "states_shape":list(states.shape),"options":enc["options"],"spans":[[a,b] for a,b in zip(enc["option_span_start"],enc["option_span_end"])],"decision_index":enc["decision_index"],
 "main_logits":z.tolist(),"main_calibrated_probabilities":p.tolist(),
 "aux48_logits":z48.tolist(),"aux48_calibrated_probabilities":p48.tolist(),
 "aux56_logits":z56.tolist(),"aux56_calibrated_probabilities":p56.tolist(),
 "finite":bool(torch.isfinite(z).all() and torch.isfinite(z48).all() and torch.isfinite(z56).all()),
 "probability_sums":[float(p.sum()),float(p48.sum()),float(p56.sum())],
},ensure_ascii=False,sort_keys=True))

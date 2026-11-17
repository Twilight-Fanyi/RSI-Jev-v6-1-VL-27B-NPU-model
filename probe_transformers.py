import sys, json, time
sys.path.insert(0,"/workspace/jev-adaptation/model-44/source-rsi-jev")
import torch_npu
import torch
print("IMPORT",torch.__version__,torch.npu.is_available(),flush=True)
from serve.release import load_release
print("LOAD_START",flush=True)
model,tok,enc,meta=load_release("/workspace/jev-adaptation/model-44/weights",device="npu:0",infer_dtype=torch.bfloat16,vision=False)
print("LOAD_OK",meta.get("serving"),flush=True)
from serve.server import Served, make_scorer
s=Served(model,tok,enc,meta,"npu:0","bf16",None,"rsi44","rsi44",None,[])
sc=make_scorer(s,batch_size=1)
print("SCORE_START",flush=True)
print(sc("I was charged twice. Please refund.", {"refund":{"type":"noul","instructions":"Does the user request a refund?"}}),flush=True)

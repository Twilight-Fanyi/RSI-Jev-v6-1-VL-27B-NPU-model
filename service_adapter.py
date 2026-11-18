from __future__ import annotations
import json, os, sys, threading
from pathlib import Path
import torch

MODEL_ID="shgao/rsi-jev-v6.1-vl-27b"
REVISION="d0dccc052ee844ab7481ddfacad1b7335ab394e0"
ARCH="Qwen3_5ForConditionalGeneration"

class Adapter:
    def __init__(self, model_dir: Path, output_dir: Path, device: str):
        if device != "npu":
            raise ValueError("model44 service requires vLLM-Ascend NPU")
        self.model_dir=Path(model_dir).resolve(strict=True)
        self.output_dir=Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.package_root=Path(__file__).resolve().parent
        # The public package intentionally omits checkpoint tensors.  Prefer a
        # complete local probe tree when one is supplied, otherwise load the
        # externally mounted model directory passed by the evaluator.
        probe_candidate=self.package_root/"vllm-base-probe"
        if (probe_candidate/"config.json").is_file() and (probe_candidate/"model.safetensors.index.json").is_file():
            self.probe_model=probe_candidate
        else:
            self.probe_model=self.model_dir
        weight_candidates=(self.model_dir/"weights",self.model_dir,self.package_root/"weights")
        self.weights_dir=next((p for p in weight_candidates if all((p/n).is_file() for n in ("scorer.safetensors","aux_scorers.safetensors","calibration.safetensors"))),None)
        if self.weights_dir is None:
            raise FileNotFoundError("scorer.safetensors, aux_scorers.safetensors and calibration.safetensors are required in --model-dir/weights")
        self.tokenizer_dir=self.package_root/"source-audit"
        self.hidden_jsonl=self.output_dir/"service-hidden.jsonl"
        self.hidden_pt=self.output_dir/"service-hidden.pt"
        os.environ["JEV_CAPTURE_HIDDEN"]="1"
        os.environ["JEV_HIDDEN_PATH"]=str(self.hidden_jsonl)
        os.environ["JEV_HIDDEN_TENSOR_PATH"]=str(self.hidden_pt)
        sys.path.insert(0,str(self.package_root/"native-plugin"))
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from jev_hidden_state_adapter import encode_option_prompt, HiddenStateBatch, score, probabilities
        from jev_scorer import load_bundle
        self.encode_option_prompt=encode_option_prompt
        self.HiddenStateBatch=HiddenStateBatch
        self.score=score
        self.probabilities=probabilities
        self.SamplingParams=SamplingParams
        self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_dir,local_files_only=True,trust_remote_code=True)
        self.main,self.aux,self.cal=load_bundle(self.weights_dir)
        self.engine=LLM(model=str(self.probe_model),tokenizer=str(self.tokenizer_dir),trust_remote_code=True,
                        tensor_parallel_size=1,dtype="bfloat16",max_model_len=2048,max_num_seqs=1,
                        enforce_eager=True,gpu_memory_utilization=0.90)
        self._lock=threading.Lock()
        self._clear_hidden()

    def _clear_hidden(self):
        for p in (self.hidden_jsonl,self.hidden_pt):
            try: p.unlink()
            except FileNotFoundError: pass

    def info(self):
        return {"model":MODEL_ID,"revision":REVISION,"model_name":"RSI-Jev v6.1 VL 27B",
                "backend":"vLLM-Ascend","architecture":ARCH,"dtype":"bfloat16",
                "tensor_parallel_size":1,"decision_types":["noul","choice","score"],
                "probabilities":"full calibrated option probability map","weights":"original pinned safetensors"}

    def _run_question(self,state,q):
        qtype=q.get("type","noul")
        if qtype=="noul":
            options=["yes","no"]
            criteria=q.get("criteria") or {"yes":"yes","no":"no"}
        elif qtype=="choice":
            criteria=q.get("criteria")
            if isinstance(criteria,dict): options=list(criteria.keys())
            elif isinstance(criteria,list): options=list(criteria); criteria={x:x for x in options}
            else: raise ValueError("choice criteria must be object or list")
        elif qtype=="score":
            criteria=q.get("criteria")
            options=list(criteria) if isinstance(criteria,list) else list((criteria or {}).keys())
            if not options: options=[str(i) for i in range(6)]
            criteria={str(k):str(v) for k,v in (criteria.items() if isinstance(criteria,dict) else [(x,x) for x in options])}
        else: raise ValueError(f"unsupported question type {qtype}")
        enc=self.encode_option_prompt(self.tokenizer,str(state or ""),str(q.get("instructions","")),options,criteria)
        self._clear_hidden()
        prompt={"prompt_token_ids":enc["input_ids"]}
        self.engine.generate([prompt],self.SamplingParams(temperature=0.0,max_tokens=1))
        if not self.hidden_pt.exists():
            raise RuntimeError("hidden capture file missing after vLLM request")
        states=torch.load(self.hidden_pt,map_location="cpu",weights_only=True)
        if states.ndim==2: states=states.unsqueeze(0)
        batch=self.HiddenStateBatch(states,torch.tensor([enc["decision_index"]]),torch.tensor([enc["option_span_start"]]),torch.tensor([enc["option_span_end"]]),torch.tensor([enc["option_mask"]],dtype=torch.bool))
        z=self.score(batch,self.main)
        calibration_path=self.weights_dir/"calibration.json"
        if calibration_path.is_file():
            logt=float(json.load(open(calibration_path))["main_logT"])
        else:
            logt=float(self.cal["main_logT"])
        p=self.probabilities(z/torch.exp(torch.tensor(logt)),batch.option_mask)[0].detach().cpu().tolist()
        probs={str(k):float(v) for k,v in zip(options,p)}
        chosen=options[max(range(len(options)),key=lambda i:p[i])]
        ans={"probabilities":probs,"choice":chosen,"decision":chosen}
        if qtype=="score":
            ans["score"]=chosen
        return ans

    def decide(self,request):
        answers={}
        state=request.get("state","")
        for qid,q in request["questions"].items():
            answers[str(qid)]=self._run_question(state,q)
        return {"model":request.get("model",MODEL_ID),"answers":answers,"usage":{"output_tokens":0}}

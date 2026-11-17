import json, sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, "/workspace/jev-adaptation/model-44/native-plugin")
from jev_hidden_state_adapter import encode_option_prompt

def main():
    tok=AutoTokenizer.from_pretrained("/workspace/jev-adaptation/model-44/source-audit", local_files_only=True, trust_remote_code=True)
    enc=encode_option_prompt(
        tok,
        "The user says the file is missing.",
        "Does the user request keeping the file?",
        ["yes","no"],
        {"yes":"the request is to keep it","no":"the request is not to keep it"},
    )
    with open("/workspace/jev-adaptation/model-44/evidence/C-real-decision-encoding.json","w",encoding="utf-8") as f:
        json.dump(enc,f,ensure_ascii=False,sort_keys=True)
    llm=LLM(
        model="/workspace/jev-adaptation/model-44/vllm-base-probe",
        tokenizer="/workspace/jev-adaptation/model-44/source-audit",
        trust_remote_code=True,tensor_parallel_size=1,dtype="bfloat16",
        max_model_len=128,max_num_seqs=1,enforce_eager=True,gpu_memory_utilization=0.90,
    )
    out=llm.generate([{"prompt_token_ids":enc["input_ids"]}], SamplingParams(temperature=0.0,max_tokens=1))
    print("REAL_DECISION_GENERATE", json.dumps({
        "input_len":len(enc["input_ids"]),
        "spans":list(zip(enc["option_span_start"],enc["option_span_end"])),
        "decision_index":enc["decision_index"],
        "token_ids":out[0].outputs[0].token_ids,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

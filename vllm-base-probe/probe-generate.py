import json
from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="/workspace/jev-adaptation/model-44/vllm-base-probe",
        tokenizer="/workspace/jev-adaptation/model-44/source-audit",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = llm.generate([{"prompt_token_ids": [0, 0, 0]}], params)
    print("GENERATE_OUTPUT", json.dumps([
        {"text": o.outputs[0].text, "token_ids": o.outputs[0].token_ids}
        for o in outputs
    ], ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

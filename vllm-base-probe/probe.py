from vllm import LLM, SamplingParams

def main():
 m=LLM(model="/workspace/jev-adaptation/model-44/vllm-base-probe",tokenizer="/workspace/jev-adaptation/model-44/source-audit",trust_remote_code=True,tensor_parallel_size=2,dtype="bfloat16",max_model_len=256,max_num_seqs=1,enforce_eager=True,gpu_memory_utilization=0.8)
 print("LOADED",flush=True)
 print(m.generate(["hello"],SamplingParams(max_tokens=1),use_tqdm=False)[0].outputs[0].text,flush=True)
if __name__=="__main__": main()

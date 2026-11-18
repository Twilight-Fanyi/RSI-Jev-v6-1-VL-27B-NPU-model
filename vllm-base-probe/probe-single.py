from vllm import LLM

def main():
    m=LLM(model="/workspace/jev-adaptation/model-44/vllm-base-probe",tokenizer="/workspace/jev-adaptation/model-44/source-audit",trust_remote_code=True,tensor_parallel_size=1,dtype="bfloat16",max_model_len=128,max_num_seqs=1,enforce_eager=True,gpu_memory_utilization=0.90)
    print("SINGLE_LOADED",flush=True)
if __name__ == "__main__": main()

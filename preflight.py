import multiprocessing

def main():
 from vllm import LLM
 print("PREFLIGHT_START_44",flush=True)
 LLM(model="/workspace/jev-adaptation/model-44/weights",tokenizer="/workspace/jev-adaptation/model-44/source-audit",trust_remote_code=True,tensor_parallel_size=2,dtype="bfloat16",max_model_len=1024,enforce_eager=True)
 print("ENGINE_READY_44",flush=True)
if __name__=="__main__": multiprocessing.set_start_method("spawn",force=True); main()

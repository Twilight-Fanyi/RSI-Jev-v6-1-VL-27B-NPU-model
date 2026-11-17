import json, sys
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/jev-adaptation/model-44/native-plugin")
from jev_hidden_state_adapter import encode_option_prompt

tok=AutoTokenizer.from_pretrained("/workspace/jev-adaptation/model-44/source-audit", local_files_only=True, trust_remote_code=True)
out=encode_option_prompt(
    tok,
    "The user says the file is missing.",
    "Does the user request keeping the file?",
    ["yes","no"],
    {"yes":"the request is to keep it","no":"the request is not to keep it"},
)
print(json.dumps({
  "tokenizer":type(tok).__name__,
  "vocab_size":len(tok),
  "input_len":len(out["input_ids"]),
  "decision_index":out["decision_index"],
  "spans":[[a,b] for a,b in zip(out["option_span_start"],out["option_span_end"])],
  "option_mask":out["option_mask"],
  "option_token_windows":[out["input_ids"][a:b] for a,b in zip(out["option_span_start"],out["option_span_end"])],
}, ensure_ascii=False, sort_keys=True))

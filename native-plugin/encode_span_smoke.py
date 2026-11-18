import json, sys, torch
sys.path.insert(0, "/workspace/jev-adaptation/model-44/native-plugin")
from jev_hidden_state_adapter import encode_option_prompt

class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 97 for ch in text]}

tok=DummyTokenizer()
out=encode_option_prompt(tok, "state", "question", ["A","B"], {"A":"yes","B":"no"})
spans=list(zip(out["option_span_start"], out["option_span_end"]))
assert spans[0][0] < spans[0][1] <= spans[1][0] < spans[1][1] <= out["decision_index"]
print(json.dumps({"input_len":len(out["input_ids"]),"spans":spans,"decision_index":out["decision_index"],"option_mask":out["option_mask"],"finite":True},sort_keys=True))

try:
    import triton
    if not hasattr(triton, "next_power_of_2"):
        def _next_power_of_2(x):
            x = int(x)
            return 1 if x <= 1 else 1 << (x - 1).bit_length()
        triton.next_power_of_2 = _next_power_of_2
    print("TRITON_PATCH", getattr(triton, "__file__", None), hasattr(triton, "next_power_of_2"), flush=True)
except Exception as e:
    print("TRITON_PATCH_ERR", repr(e), flush=True)

from vllm.model_executor.models.utils import WeightsMapper
P={
    "embed_tokens.": "language_model.model.embed_tokens.",
    "layers.": "language_model.model.layers.",
    "norm.": "language_model.model.norm.",
    "blocks.": "visual.blocks.",
    "merger.": "visual.merger.",
    "patch_embed.": "visual.patch_embed.",
    "pos_embed.": "visual.pos_embed.",
}
for clsname, modname in [
    ("Qwen3_5ForConditionalGeneration", "vllm.model_executor.models.qwen3_5"),
    ("AscendQwen3_5ForConditionalGeneration", "vllm_ascend.models.qwen3_5_vl"),
]:
    try:
        mod=__import__(modname, fromlist=[clsname]); cls=getattr(mod, clsname)
        cls.hf_to_vllm_mapper.orig_to_new_prefix.update(P)
        print("PATCH_OK", clsname, cls.hf_to_vllm_mapper.orig_to_new_prefix, flush=True)
    except Exception as e:
        print("PATCH_ERR", clsname, repr(e), flush=True)

try:
    import vllm.triton_utils as _vllm_triton_utils
    _vtu = _vllm_triton_utils.triton
    if not hasattr(_vtu, "next_power_of_2"):
        def _vtu_next_power_of_2(x):
            x = int(x)
            return 1 if x <= 1 else 1 << (x - 1).bit_length()
        _vtu.next_power_of_2 = _vtu_next_power_of_2
    print("VLLM_TRITON_PATCH", type(_vtu).__name__, hasattr(_vtu, "next_power_of_2"), flush=True)
except Exception as e:
    print("VLLM_TRITON_PATCH_ERR", repr(e), flush=True)

try:
    import os as _jev_os
    if _jev_os.environ.get("JEV_CAPTURE_HIDDEN") == "1":
        _jev_hidden_path = _jev_os.environ.get("JEV_HIDDEN_PATH", "/workspace/jev-adaptation/model-44/evidence/hidden-states.jsonl")
        from vllm.model_executor.models.qwen3_next import Qwen3NextModel as _JevQwen3NextModel
        _JevOrigForward = _JevQwen3NextModel.forward
        def _JevCaptureForward(self, *args, **kwargs):
            _JevOut = _JevOrigForward(self, *args, **kwargs)
            if getattr(self, "_jev_hidden_records", 0) < 8:
                _JevT = _JevOut[0] if isinstance(_JevOut, (tuple, list)) else _JevOut
                if hasattr(_JevT, "detach"):
                    try:
                        _JevFlat = _JevT.detach().reshape(-1)[:4096].float().cpu()
                        _JevRec = {
                            "shape": list(_JevT.shape),
                            "mean": float(_JevFlat.mean().item()),
                            "std": float(_JevFlat.std().item()),
                            "finite": bool(_JevFlat.isfinite().all().item()),
                        }
                        if _JevRec["std"] > 1e-3 and not _jev_os.environ.get("JEV_HIDDEN_TENSOR_PATH", "") == "":
                            try:
                                import torch as _JevTorch
                                _JevTorch.save(_JevT.detach().float().cpu(), _jev_os.environ["JEV_HIDDEN_TENSOR_PATH"])
                                _JevRec["tensor_path"] = _jev_os.environ["JEV_HIDDEN_TENSOR_PATH"]
                            except Exception as _JevSaveE:
                                print("JEV_HIDDEN_TENSOR_SAVE_ERR", repr(_JevSaveE), flush=True)
                        with open(_jev_hidden_path, "a", encoding="utf-8") as _JevF:
                            import json as _JevJson
                            _JevF.write(_JevJson.dumps(_JevRec, ensure_ascii=False) + "\n")
                        self._jev_hidden_records = getattr(self, "_jev_hidden_records", 0) + 1
                    except Exception as _JevE:
                        print("JEV_HIDDEN_CAPTURE_ERR", repr(_JevE), flush=True)
            return _JevOut
        _JevQwen3NextModel.forward = _JevCaptureForward
        print("JEV_HIDDEN_FORWARD_PATCH", _jev_hidden_path, flush=True)
except Exception as _JevE:
    print("JEV_HIDDEN_FORWARD_SETUP_ERR", repr(_JevE), flush=True)

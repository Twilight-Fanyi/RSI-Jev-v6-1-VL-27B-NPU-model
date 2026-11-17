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

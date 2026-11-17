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

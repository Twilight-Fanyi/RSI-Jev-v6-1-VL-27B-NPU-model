# model-44 release status

- **Pinned source:** `shgao/rsi-jev-v6.1-vl-27b@d0dccc052ee844ab7481ddfacad1b7335ab394e0`.
- **Runtime evidence:** vLLM-Ascend 0.30.0, CANN 9.2.1, Ascend910/A3, BF16; TP2 loader probe and TP1 chip1 service run are retained.
- **Decision path:** real tokenizer + real NPU hidden capture + typed scorer/calibration + four HTTP routes.
- **Replay parity:** `evidence/C-http-precision.json` reports max absolute NPU/CPU scorer replay delta `0.0`.
- **Official baseline:** not required as a numeric 2PP gate for the adaptation track; no official score is claimed.
- **Weights:** excluded from this public package. Use the exact pinned model directory and verify it against `lfs-manifest.json` before running.
Screenshots: complete; `assets/agent_workflow.png`, `assets/npu_device_call.png`, and `assets/model_result.png` are live model-44 terminal captures. Prior model-102 PNGs remain quarantined under `evidence/legacy-model102-assets/`.

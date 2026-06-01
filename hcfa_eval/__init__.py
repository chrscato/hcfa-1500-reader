"""Fine-tuning dataset + evaluation harness for HCFA-1500 extraction.

Built around the synthetic data emitted by `hcfa_synth`. Targets Qwen2.5-VL
(and any other vision-LM that accepts chat-style messages with embedded
images), but the scoring core is model-agnostic — it scores any predictions
file that emits {sample_id, fields} rows.
"""

__version__ = "0.1.0"

TIER_NAMES = ["pristine", "clean_scan", "worn_scan", "fax", "phone_photo", "worst"]

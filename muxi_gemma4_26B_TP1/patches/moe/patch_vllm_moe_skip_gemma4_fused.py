"""Skip TransformersFusedMoE replacement when experts use Gemma4-style gate_up_proj packing."""
from pathlib import Path

p = Path("/opt/conda/lib/python3.10/site-packages/vllm/model_executor/models/transformers/moe.py")
text = p.read_text()
bak = p.with_suffix(".py.bak.skip_gemma4_fused")
if not bak.exists():
    bak.write_text(text)

old = """                if child_name == "experts" and (is_modulelist or is_3d):
                    # Alias for readability
                    mlp = module
                    experts = child_module"""

new = """                if child_name == "experts" and (is_modulelist or is_3d):
                    # Gemma4 uses fused gate_up_proj + down_proj tensors; FusedMoE's
                    # expert_mapping expects Mixtral-style per-shard names and will
                    # not load these checkpoints. Keep native Transformers experts.
                    if any(
                        "gate_up_proj" in n
                        for n, _ in child_module.named_parameters()
                    ):
                        _recursive_replace(child_module, prefix=qual_name)
                        continue
                    # Alias for readability
                    mlp = module
                    experts = child_module"""

if old not in text:
    raise SystemExit("anchor not found")
p.write_text(text.replace(old, new, 1))
print("patched ok")

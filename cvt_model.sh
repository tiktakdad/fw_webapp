#/bin/bash
python scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors --extract_ema --checkpoint_path "model/base.safetensors" --dump_path "model/dump" --device cuda:0
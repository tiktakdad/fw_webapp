#/bin/bash
python convert_original_stable_diffusion_to_diffusers.py --from_safetensors --checkpoint_path "../model/AOM3A3_base.safetensors" --dump_path "../model/dump" --device cuda:0
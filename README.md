# fw_webapp


## quick start
```shell
(optional) conda create -n fw_webapp python=3.9
(optional) conda activate fw_webapp

git clone https://github.com/tiktakdad/fw_webapp.git
cd fw_webapp
pip install -r requirements.txt
python app.py
```

## convert custom sd model
- cvt_model.sh
```shell
python scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors --extract_ema --checkpoint_path "model/base.safetensors" --dump_path "model/dump" --device cuda:0
```

## sd model download
- copy to fw_webapp/model
- https://1drv.ms/u/s!AgGeqTkQEyKRib8iuOxTmLwXERrenA?e=nQyTEh

## clip model download
- copy to fw_webapp/BLIP/pth
- https://1drv.ms/f/s!AgGeqTkQEyKRib8j7ICWFR1JU8KpmA?e=agVGJ0

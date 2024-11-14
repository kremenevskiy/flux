```
apt-get install ffmpeg libsm6 libxext6  -y
pip install diffusers==0.30.2 opencv-python uvicorn fastapi torch huggingface_hub transformers sentencepiece accelerate python-multipart protobuf jupyter lab
huggingface-cli login --token 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
```


### Setup commands


```
git lfs install
```

Canny edge filter
```
git clone https://huggingface.co/spaces/Deadmon/FLUX.1-DEV-Canny flux-canny
python -m venv flux
source flux/bin/activate
cd flux-canny
pip install --upgrade pip
pip install -r requirements.txt
# comment spaces
huggingface-cli login --token $HF_TOKEN
huggingface-cli login --token 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
python app.py
```


Working inpainting low gpu
git clone https://github.com/alimama-creative/FLUX-Controlnet-Inpainting.git inpaint

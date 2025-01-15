```
apt-get install ffmpeg libsm6 libxext6 htop nvtop -y &&
python -m venv venv &&
. venv/bin/activate &&
pip install --upgrade pip && 
pip install diffusers==0.30.2 opencv-python uvicorn fastapi torch huggingface_hub transformers sentencepiece accelerate python-multipart protobuf einops pandas jupyter lab gradio gradio_imageslider peft && 
huggingface-cli login --token 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
```

## RUN:
```
nohup python service.py > service.log 2>&1 &
```


```
apt-get install ffmpeg libsm6 libxext6 htop nvtop -y &&
python -m venv venv &&
. venv/bin/activate &&
pip install --upgrade pip && 
pip install -r requirements.txt
huggingface-cli login --token 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
```

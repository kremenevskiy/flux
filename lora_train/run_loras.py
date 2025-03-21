import os
import subprocess
import sys

# List of config files
config_files = [
    'flux/lora_train/lora_configs/styles_fast_sergey_dataset_rank_8_lr5_4.yml',
    'flux/lora_train/lora_configs/styles_fast_sergey_dataset_rank_8_lr5_5.yml',
    'flux/lora_train/lora_configs/styles_fast_sergey_dataset_rank_8_lr5_4_full.yml',
    'flux/lora_train/lora_configs/styles_fast_sergey_dataset_rank_8_lr5_5.yml',
]


# Command template
command_template = ['python', 'ai-toolkit/run.py']

existing_configs = [config for config in config_files if os.path.exists(config)]

if not existing_configs:
    print('❌ No valid config files found! Exiting.')
    sys.exit(1)

# Run each training process sequentially
for config in config_files:
    print(f'Starting training with: {config}')
    process = subprocess.run(command_template + [config], check=True)
    print(f'Finished training with: {config}\n')

print('All trainings completed!')

import os
import subprocess

# List of commands to run sequentially


config_files = [
    # 'flux/lora_train/lora_configs/styles_latest_d.yml',
    'flux/lora_train/lora_configs/styles_latest_b.yml',
    'flux/lora_train/lora_configs/styles_latest_c.yml',
]

for config_file in config_files:
    if not os.path.exists(config_file):
        print(f'Config file {config_file} does not exist')
        raise Exception(f'Config file {config_file} does not exist')

commands = ['python ai-toolkit/run.py ' + config_file for config_file in config_files]


# Run each command one by one
for cmd in commands:
    print(f'Running: {cmd}')
    process = subprocess.run(cmd, shell=True)
    if process.returncode != 0:
        print(f'Error: Command failed - {cmd}')
        break  # Stop execution if any command fails

print('All processes completed.')

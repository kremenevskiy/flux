import subprocess

# List of commands to run sequentially
commands = [
    'python ai-toolkit/run.py ai-toolkit/config/examples/styles_a.yml',
    'python ai-toolkit/run.py ai-toolkit/config/examples/styles_b.yml',
    'python ai-toolkit/run.py ai-toolkit/config/examples/styles_c.yml',
    'python ai-toolkit/run.py ai-toolkit/config/examples/styles_d.yml',
    'python ai-toolkit/run.py ai-toolkit/config/examples/styles_e.yml',
]

# Run each command one by one
for cmd in commands:
    print(f'Running: {cmd}')
    process = subprocess.run(cmd, shell=True)
    if process.returncode != 0:
        print(f'Error: Command failed - {cmd}')
        break  # Stop execution if any command fails

print('All processes completed.')

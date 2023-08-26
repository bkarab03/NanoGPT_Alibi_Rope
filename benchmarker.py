import subprocess
import time

# List of training commands to run sequentially
training_commands = [
    "python train.py config/train_shakespeare_char.py --compile=False --block_size=128 --batch_size=12 --n_layer=6 --n_head=4 --n_embd=128 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --init_from=scratch --eval_interval=1000 --model_type=GPT --max_time_minutes=1"
    # Add more commands here as needed
]

# Execute each command sequentially
for cmd in training_commands:
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True)
    print("Command finished, moving to the next one.")
    time.sleep(10)  # Optional: sleep for 2 seconds between commands

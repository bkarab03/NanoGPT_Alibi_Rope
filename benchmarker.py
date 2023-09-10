import subprocess
import os
import matplotlib.pyplot as plt
import json

model_type = "GPT"
n_embd = "1024"
block_size = "1024"
time_min = "3"

training_commands = [
    f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal"
    # f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal --pos_enc_type=alibi",
    # f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal --pos_enc_type=rope"
]

for cmd in training_commands:
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True)
    print("Command finished, moving to the next one.")

# Get all filenames in the current directory that start with 'loss_data'
chart_dir = 'charts'
all_filenames = os.listdir(chart_dir)
filenames = [os.path.join(chart_dir, fname) for fname in all_filenames if fname.startswith('loss_data')]

plt.figure(figsize=(10, 6))

for fname in filenames:
    with open(fname, 'r') as f:
        losses = json.load(f)
    plt.plot(losses, label=fname)  # Use some part of the filename as the label

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Comparing Training Losses')
plt.legend()
plt.savefig('combined_loss_chart.png')








# training_commands = [
#     f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=retention",
#     f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=3 --init_from=scratch --eval_interval=10 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal"
#     # f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=grouped"
# ]

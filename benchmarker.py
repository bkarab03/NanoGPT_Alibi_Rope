import tkinter as tk
from tkinter import messagebox
import os
import matplotlib.pyplot as plt
import json
import subprocess
import sys  # Import sys at the beginning of your script
import shutil  # Import shutil for directory operations

# Function to clear the contents of the charts directory
def clear_chart_dir(chart_dir):
    if os.path.exists(chart_dir):
        for file in os.listdir(chart_dir):
            file_path = os.path.join(chart_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Function to run the training commands
def run_training_commands():
    for cmd in training_commands:
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True)
        print("Command finished, moving to the next one.")

# Function to plot the loss
def plot_loss(chart_dir):
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)

    filenames = [os.path.join(chart_dir, fname) for fname in os.listdir(chart_dir) if fname.startswith('loss_data')]
    plt.figure(figsize=(10, 6))

    for fname in filenames:
        with open(fname, 'r') as f:
            losses = json.load(f)
        plt.plot(losses, label=fname)  # Use some part of the filename as the label

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Comparing Training Losses')
    plt.legend()
    plt.savefig(os.path.join(chart_dir, 'combined_loss_chart.png'))
    plt.show()

# Function to update parameters from the UI and run training
def update_parameters_and_train():
    global model_type, n_embd, block_size, time_min, max_iters, training_commands

    # Update parameters with values from the entry fields
    model_type = model_type_entry.get()
    n_embd = n_embd_entry.get()
    block_size = block_size_entry.get()
    time_min = time_min_entry.get()
    max_iters = max_iters_entry.get()
    chart_dir = 'charts'

    # Update the training commands with the new parameters
    training_commands = [
        f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters={max_iters} --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal",
        f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters={max_iters} --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal --pos_enc_type=alibi"
        # f"python train.py config/train_shakespeare_char.py --compile=False --block_size={block_size} --batch_size=12 --n_layer=6 --n_head=4 --n_embd={n_embd} --max_iters={max_iters} --lr_decay_iters=2000 --dropout=0.0 --eval_iters=2 --init_from=scratch --eval_interval=1000 --model_type={model_type} --max_time_minutes={time_min} --attention_type=causal --pos_enc_type=rope"        # Add more commands as needed
    ]

    # Now we can run the training commands
    run_training_commands()

    # After training, plot the loss
    plot_loss(chart_dir)

    # Show a message box that training is done
    messagebox.showinfo("Training", "Training has been completed.")
# Function to handle quit action
def on_quit():
    if messagebox.askyesno("Quit", "Do you really wish to quit?"):
        root.destroy()  # Destroy the Tkinter window
        sys.exit()      # Terminate the Python process

chart_dir = 'charts'
clear_chart_dir(chart_dir)

# Set up the Tkinter UI
root = tk.Tk()
root.title("Training Parameters Input")

# Entry fields for parameters
tk.Label(root, text="Model Type").grid(row=0, column=0)
model_type_entry = tk.Entry(root)
model_type_entry.grid(row=0, column=1)
model_type_entry.insert(0, "GPT")

tk.Label(root, text="N Embedding").grid(row=1, column=0)
n_embd_entry = tk.Entry(root)
n_embd_entry.grid(row=1, column=1)
n_embd_entry.insert(0, "256")

tk.Label(root, text="Block Size").grid(row=2, column=0)
block_size_entry = tk.Entry(root)
block_size_entry.grid(row=2, column=1)
block_size_entry.insert(0, "256")

tk.Label(root, text="Time (min)").grid(row=3, column=0)
time_min_entry = tk.Entry(root)
time_min_entry.grid(row=3, column=1)
time_min_entry.insert(0, "1")

tk.Label(root, text="Max Iterations").grid(row=4, column=0)
max_iters_entry = tk.Entry(root)
max_iters_entry.grid(row=4, column=1)
max_iters_entry.insert(0, "20")

# Button to start training
start_button = tk.Button(root, text="Start Training", command=update_parameters_and_train)
start_button.grid(row=5, column=0, columnspan=2)

# Quit button
quit_button = tk.Button(root, text="Quit", command=on_quit)
quit_button.grid(row=6, column=0, columnspan=2)

# Start the Tkinter event loop
root.mainloop()

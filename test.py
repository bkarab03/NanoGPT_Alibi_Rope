import subprocess

encodings = ['alibi', 'original']
command_template = "python train.py config/train_shakespeare_char.py --compile=False --log_interval=20 --block_size=128 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=3000 --lr_decay_iters=2000 --dropout=0.0 --pos_enc_type={} --max_time=35"

for encoding in encodings:
    cmd = command_template.format(encoding)
    subprocess.run(cmd, shell=True)

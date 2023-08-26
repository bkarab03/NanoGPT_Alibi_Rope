import json
import random
import torch
import os
import requests
from tqdm import tqdm

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def get_tokenizer():
    return tokenizer

def download_file(url, file_path):
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(file_path, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()


base_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
train_dataset = 'dev-v2.0.json'
eval_dataset = 'dev-v2.0.json'

import pickle

def prepare_data(dataset):
    file_path = os.path.join(os.path.dirname(__file__), dataset)
    if not os.path.exists(file_path):
        download_file(base_url + dataset, file_path)
    with open(file_path, 'r') as f:
        squad_dict = json.load(f)

    contexts = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            contexts.append(context)

    random.shuffle(contexts)
    input_ids = []
    masks = []
    target_ids = []

    inputs_and_masks = []
    for i in range(len(contexts) - 1):
        inputs = tokenizer([contexts[i]], [contexts[i + 1]], truncation=True, max_length=128, padding='max_length')
        target_ids = inputs['input_ids'].copy()
        probability_matrix = torch.full((1, len(inputs['input_ids'][0])), 0.15)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs['input_ids']
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs['input_ids'] = torch.tensor(inputs['input_ids'])
        inputs['input_ids'][masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        inputs['mask'] = masked_indices.int()
        inputs['target_ids'] = torch.tensor(target_ids)
        inputs_and_masks.append(inputs)

    # Save inputs_and_masks using torch.save
    torch_save_file = f"{os.path.dirname(__file__)}/{dataset.split('.')[0]}_inputs_and_masks.pt"
    torch.save(inputs_and_masks, torch_save_file)

    return inputs_and_masks


def print_items(inputs_and_masks, num_items=1):
    for item in inputs_and_masks[:num_items]:
        print("Original sequence:")
        for seq in item['target_ids']:
            print(tokenizer.decode(seq, skip_special_tokens=False))
        print("Input sequence (masked):")
        for seq in item['input_ids']:
            print(tokenizer.decode(seq, skip_special_tokens=False))
        # print("mask:")
        # print(item['mask'])

inputs_and_masks_train = prepare_data(train_dataset)
# inputs_and_masks_eval = prepare_data(eval_dataset)

print_items(inputs_and_masks_train)





import os
import json
from transformers import AutoTokenizer
from transformers import LineByLineTextDataset

import torch
from torch import nn
from reformer_pytorch import ReformerLM

from electra_pytorch import Electra
from dataset import *


DATA_DIR = "."  #@param {type: "string"}
TRAIN_SIZE = 11000  #@param {type:"integer"}
MODEL_NAME = "electra-semeval2023" #@param {type: "string"}   


print('loading tokenizer ...')

# Save the pretrained WordPiece tokenizer to get `vocab.txt`
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained(DATA_DIR)

print('making dataset ...')
ds_name="unlabeled_data.txt"
# Convert data into required format
ds_command=f'python3 build_pretraining_dataset.py   --corpus-dir {DATA_DIR}  --vocab-file {DATA_DIR}/vocab.txt  --output-dir {DATA_DIR}/pretrain_tfrecords  --max-seq-length 128   --blanks-separate-docs False   --no-lower-case   --num-processes 5'
d=os.popen(ds_command).read()

"""print('refactoring dataset with one sample ...')
with open(file='train_data.txt' , mode='r',encoding='utf-8') as f:
    text=f.readline()
    f.close()

with open(file=f'train_data.txt' , mode='w',encoding='utf-8') as f:
    f.write(text)
    f.write('\n')
    f.close()
    """

print('forming training hparams ...')

hparams = {
    "output_dir":f'./output_electra',
    "model_dir":'output_electra',
    "overwrite_output_dir":True,
    "do_train": "true",
    "do_eval": "false",
    "model_size": "small",
    "do_lower_case": "false",
    "vocab_size": 119547,
    "num_train_steps": 100,
    "num_train_epochs":50,
    "save_checkpoints_steps": 100,
    "save_steps":10_000,
    "train_batch_size": 32,
    "save_total_limit":3,
    "learning_rate":1e-5,
    "resume_from_checkpoint":True,
    "generator_hidden_size": 1.0,
}

print('saving hparams ...')

with open("hparams.json", "w") as f:
    json.dump(hparams, f)

# (1) instantiate the generator and discriminator, making sure that the generator is roughly a quarter to a half of the size of the discriminator

generator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 256,              # smaller hidden dimension
    heads = 4,              # less heads
    ff_mult = 2,            # smaller feed forward intermediate dimension
    dim_head = 64,
    depth = 12,
    max_seq_len = 1024
)

discriminator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    dim_head = 64,
    heads = 16,
    depth = 12,
    ff_mult = 4,
    max_seq_len = 1024,
    return_embeddings = True
)

# (2) weight tie the token and positional embeddings of generator and discriminator

generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb
# weight tie any other embeddings if available, token type embeddings, etc.

# (3) instantiate electra

discriminator_with_adapter = nn.Sequential(discriminator, nn.Linear(1024, 1))

trainer = Electra(
    generator,
    discriminator_with_adapter,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    mask_ignore_token_ids = []  # ids of tokens to ignore for mask modeling ex. (cls, sep)
)

data=OpenWebTextDataset()
print('running pretraining ...')
results = trainer(data)
results.loss.backward()

# after much training, the discriminator should have improved

torch.save(discriminator, f'./pretrained-model.pt')
command=f'python3 run_pretraining.py --data-dir {DATA_DIR}  --model-name {MODEL_NAME} --hparams "hparams.json" ' 
result=os.popen(command).read()
print(result)
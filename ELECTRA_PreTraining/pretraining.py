#python3 pip install tensorflow==1.15
#python3 pip install transformers==2.8.0
import os
import json
from transformers import AutoTokenizer
from transformers import LineByLineTextDataset

DATA_DIR = "."  #@param {type: "string"}
TRAIN_SIZE = 1000000  #@param {type:"integer"}
MODEL_NAME = "electra-semeval2023" #@param {type: "string"}   


print('loading tokenizer ...')

# Save the pretrained WordPiece tokenizer to get `vocab.txt`
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained(DATA_DIR)

print('making dataset ...')
ds_name="unlabeled_data.txt"

print('forming training hparams ...')

hparams = { 
    "model_dir": "output_electra",  
    "do_train": "True", 
    "do_eval": "False", 
    "model_size": "small", 
    "do_lower_case": "false", 
    "vocab_size": 119547, 
    "num_train_steps": 65000, 
    "save_checkpoints_steps": 1000, 
    "train_batch_size": 32,  
    "learning_rate": 1e-05, 
    "generator_hidden_size": 1.0
    }

print('saving hparams ...')

with open("hparams.json", "w") as f:
    json.dump(hparams, f)


"""# Convert data into required format
ds_command=f'python3 build_pretraining_dataset.py   --corpus-dir {DATA_DIR}  --vocab-file {DATA_DIR}/vocab.txt  --output-dir {DATA_DIR}/pretrain_tfrecords  --max-seq-length 128   --blanks-separate-docs False   --no-lower-case   --num-processes 5'
d=os.popen(ds_command).read()

"""
"""
print('refactoring dataset with one sample ...')
with open(file='train_data.txt' , mode='r',encoding='utf-8') as f:
    text=f.readline()
    f.close()

with open(file=f'train_data.txt' , mode='w',encoding='utf-8') as f:
    f.write(text)
    f.write('\n')
    f.close()
    


print('running pretraining ...')
command=f'python3 run_pretraining.py --data-dir {DATA_DIR}  --model-name {MODEL_NAME} --hparams "hparams.json" ' 
result=os.popen(command).read()
print(result)"""
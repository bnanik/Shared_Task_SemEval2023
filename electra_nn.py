#pip install transformers
import pandas as pd
import json,re,os
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer,BertForTokenClassification,BertForSequenceClassification,BertTokenizer, BertConfig,AutoTokenizer
from transformers import TrainingArguments, Trainer ,AdamW,get_linear_schedule_with_warmup
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import ElectraConfig, ElectraModel,ElectraTokenizer,ElectraForSequenceClassification
#import tensorflow as tf
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,precision_recall_fscore_support, f1_score, precision_score,recall_score
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import time
import datetime
import random
from tqdm import tqdm

global max_len
global unseen_label,tag2idx
model_id=  "bhadresh-savani/electra-base-emotion" #  "google/electra-small-discriminator"

tokenizer = ElectraTokenizer.from_pretrained(model_id)


#%% configs

# run #22
runMessage='run#22'
num_epochs = 20
batch_size = 32
learningRate=1e-5
epsilone=1e-8
warmeup_step=0
seed_vals = [0,42,80]
es_patience=3
#Bert tokenizer do_lower_case=False
#Ber_model(bert-base-cased)

weight_decay=0.01
no_decay = ['bias', 'LayerNorm.weight']
labels__=['O','B-EVENT','I-EVENT']
global label_to_ids,ids_to_label
label_to_ids = {'O': 0,'B-EVENT': 1, 'I-EVENT': 2}
ids_to_label = {0:'O',1:'B-EVENT', 2:'I-EVENT'}
#%% main functionalities
# early stopping in number of epochs
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a <= best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a >= best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a <= best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a >= best + (
                            best * min_delta / 100)


class Ner_Data_Token(Dataset):

    def __init__(self, data):
        self.data = data
#         print("dataloader initialized")
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         print(idx)
        #print(f'sentence: id:{self.data["text"][idx]}')
        sentence = str(self.data['text'][idx])
        word_labels = [self.data['label'][idx]] #.split(" ") 
          
        sen_code = tokenizer.encode_plus(sentence,       #tokenizer.encode_plus(..)
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 64,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            truncation=True,
            #return_tensors = 'pt'
            )
             
        #print('sen_code',sen_code)    
        labels = []
        word_ids = sen_code['input_ids']
        #print(f'sentence: id:{self.data["text"][idx]} \nsen_code: {sen_code} \nword_ids:{word_ids}')
        first_token=False
        for i, label in enumerate(word_labels):
              #sen_code.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None or word_idx in [101,102]:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif  first_token==False and word_idx != previous_word_idx :
                    label_ids.append(label)#(label_to_ids[label])
                    first_token=True
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        
        
        #print('labels',labels)

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] = torch.as_tensor(labels)
        item['token']=sentence
        return item
class Ner_Data_Sentence(Dataset):

    def __init__(self, data_file):
        self.data = self.read_data(data_file)
        sents, tags_li = [], [] # list of lists
        self.labels___=[]
        for sent in self.data:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            #print(f"words: {words}  labels: {tags}")
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li

        self.labels___=self.unique([item for sublist in self.tags_li for item in sublist])
        
#         print("dataloader initialized")
        
        
    def read_data(self,filename):
        sentences = []
        with open(filename, 'r', encoding='UTF-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    sentence.append((line.split(' ')[0], line.split(' ')[-1]))
                    #sentence.append((line.split(' ')[0], line.split(' ')[-1]))
                elif len(sentence) != 0:
                    sentences.append(sentence)
                    sentence = []
        self.tags = list(set(word_label[1] for sent in sentences for word_label in sent))
        return sentences 
    def unique(self,list1):
        # initialize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    
    @property  
    def label_to_ids(self):
        return {tag:idx for idx, tag in enumerate(self.labels___)}

    
    @property
    def ids_to_label(self):
        return {idx:tag for idx, tag in enumerate(self.labels___)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        #print(f'sentence: id:{self.data["text"][idx]}')
        words, word_labels = self.sents[idx], self.tags_li[idx]
        #sentence = str(self.data['text'][idx])
        #word_labels = [self.data['label'][idx]] #.split(" ") 
        #print(f'sentence: id:{self.data["text"][idx]} \nsen_code: {sen_code} \nword_ids:{word_ids}')
        input_ids, labels = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, word_labels):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)
            if t.strip()=='':
                t='O'
            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            #print(self.label_to_ids)
            yy = [self.label_to_ids[each] for each in t]  # (T,)

            input_ids.extend(xx)
            is_heads.extend(is_head)
            labels.extend(yy)

        assert len(input_ids)==len(labels)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(input_ids), len(labels), len(is_heads))

        # seqlen
        seqlen = len(labels)

        # to string
        words = ' '.join(words)
        tags = ' '.join(word_labels)
        # step 4: turn everything into PyTorch tensors
        item = {} #{key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['input_ids']=input_ids#torch.as_tensor(input_ids)
        item['labels'] = labels #torch.as_tensor(labels)
        item['words']=words
        item['originalLabels']=tags
        item['attention_mask']=is_heads
        item['seqlen']=seqlen
        return item

def compute_metrics(pred):
    pred=pred[0]
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
def read_source(filename):
    cols=['token','label']
    tokens=[]
    labels=[]
    d=pd.read_csv(filename)
    data=pd.DataFrame(data=d,columns=cols)
    return data #data['token'],data['label']


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return  accuracy_score(labels_flat,pred_flat) #np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_scores(preds, labels):
    #pred_flat = np.argmax(preds, axis=1).flatten()
    #labels_flat = labels.flatten()
    return precision_recall_fscore_support(labels, preds, average='macro')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def saveModel(model:BertForTokenClassification,tokenizer:BertTokenizer,stats,text='run'):
    ts=time.time()
    output_dir = f'model_save/{text}'
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    training_stats_df=pd.DataFrame(data=stats)
    training_stats_df.to_csv(os.path.join(output_dir, f'training_stats_{runMessage}.csv'))


class MetricsTracking():
  """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """
  def __init__(self,title,ids_ToLabel,tknzr:BertTokenizer):

    self.total_acc = 0
    self.total_f1 = 0
    self.total_precision = 0
    self.total_recall = 0
    self.total_predictions=[]
    self.total_label=[]
    self.total_input=[]
    self.words=[]
    self.heads=[]
    self.tags=[]
    self.title=title
    self.idsToLabel=ids_ToLabel
    self.tokenizer=tknzr
    self.current_pred=[]
    self.current_tags=[]
    self.current_words=[]

  def getClassificationReport(self):
        #print("total_predictions",self.total_predictions)
        #print("total_label",self.current_tags)

        tags = list(set(w for w in self.current_tags[1:-1]))
        #print('tags',tags)
        return classification_report(self.current_tags, self.current_pred,labels=tags, zero_division=0)

  def save_results(self,predictions, labels ,inputs,epoch):
    output_dir = f'model_save/results'
    f_name=os.path.join(output_dir, f'details_{self.title}_{runMessage}.txt')
    with open(f_name, 'a') as fout:
        for idx,item  in enumerate(list(zip(inputs,labels, predictions))):
                token=item[0]
                tag=item[1]
                y_hat=item[2]
                print(f'epoch:{epoch}\n,token:{token} \n,tag: {tag}\n,y_hat: {y_hat}\n')
                fout.write(f"{epoch} {str(token)} {tag} {y_hat} \n")
                
    fout.close()    

  def update(self, predictions, labels ,inputs,words,heads,tags,epoch, ignore_token = -100):
    '''
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    '''  
    #print("start updating metrics ...")
    #predictions = predictions.flatten()
    #labels = labels.flatten()
    self.current_pred=[]
    self.current_tags=[]
    self.current_words=[]
    self.words.extend(words)
    self.heads.extend(heads)
    self.tags.extend(tags)
    n_labels=[]
    n_predictions=[]
    n_input_ids=[]
    #print('words',f'{words} tag:{tags}  pred:{predictions} labels:{labels}  heads:{heads}')
    for idx,item  in enumerate(list(zip(words,tags, predictions,heads))):

                token=item[0]
                tag=item[1]
                y_hat=item[2]
                hd=item[3]
                preds=[self.idsToLabel[x] for x,y in zip(y_hat,hd) if y==1]
                print('input token',f'{token} tag:{tag}  pred:{y_hat}')
                try:
                    assert len(preds)==len(token.split())==len(tag.split())
                    for i,row in enumerate(list(zip(token.split()[1:-1],tag.split()[1:-1],preds[1:-1]))):
                        tk=row[0]
                        tg=row[1]
                        yy_hat=row[2]
                        #if tk not in (0,101,102):
                            #print("inner loop for each token")
                        try:
                            #print(f'epoch: {epoch}')
                            #print("token:",str(tk))
                            #print("tag:",tg)
                            #print("yy_hat:",yy_hat)

                            n_labels.append(tg)
                            n_predictions.append(yy_hat)
                            n_input_ids.append(tk)
                            
                            #print(f'{epoch},{tk},{self.idsToLabel[tg]},{self.idsToLabel[yy_hat]}\n')
                            #print('{} {} \n'.format( ids_to_label[tag], ids_to_label[y_hat]))
                        except:
                            #print(f'{epoch},{self.tokenizer.convert_ids_to_tokens(tk)},{self.idsToLabel[tg]},{self.idsToLabel[yy_hat]}\n')
                            print("error")
                except:
                    print(f'ERROR!  input token',f'{token} tag:{tag}  pred:{y_hat}') 
                    continue           

    
    acc = accuracy_score(n_labels,n_predictions)
    f1 = f1_score(n_labels, n_predictions, average = "macro")
    precision = precision_score(n_labels, n_predictions, average = "macro")
    recall = recall_score(n_labels, n_predictions, average = "macro")
    
    self.total_acc  += acc
    self.total_f1 += f1
    self.total_precision += precision
    self.total_recall  += recall
    self.total_predictions.extend(n_predictions)
    self.total_label.extend(n_labels)
    self.total_input.extend(n_input_ids)
    self.current_tags.extend(n_labels)
    self.current_words.extend(n_input_ids)
    self.current_pred.extend(n_predictions)

    
    self.save_results(n_predictions,n_labels,n_input_ids,epoch)

    #print("end updating metrics ...")

  
  def return_avg_metrics(self,data_loader_size):
    n = data_loader_size
    metrics = {
        "acc": round(self.total_acc / n ,3), 
        "f1": round(self.total_f1 / n, 3), 
        "precision" : round(self.total_precision / n, 3), 
        "recall": round(self.total_recall / n, 3)
          }
    return metrics   
def padding(batch):
     
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    
    words = f('words')
    #is_heads = f('attention_mask')
    tags = f('labels')
    original_labels = f('originalLabels')
    seqlens = f('seqlen')
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f("input_ids", maxlen)
    y = f("labels", maxlen)
    z = f("attention_mask", maxlen)

    f = torch.LongTensor
    item={}
    item['words']=words
    item['tags']=tags
    item['input_ids']=f(x)
    item['labels']=f(y)
    item['attention_mask']=f(z)
    item['originalLabels']=original_labels
    item['seqlen']=seqlens
    #print(f'f(labels): {item["labels"]} \n  f(input_ids): {item["input_ids"]}')
   

    return item

def main():
    max_len=0
    # cuda avaliability?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.manual_seed(seed_val)
    configuration = ElectraConfig()
    #model definition
    model = ElectraForSequenceClassification.from_pretrained(model_id)

    configuration = model.config
    
    model.to(device)
    lastModel=model
    #preparing dataframes
    train_file='train.txt'
    dev_file='dev.txt'
    test_file='test.txt'
    """df_trainDS=pd.read_csv('baseline_train.csv')
    df_devDS=pd.read_csv('baseline_dev.csv')
    df_testDS=pd.read_csv('baseline_test.csv')"""
    
    #datasets
    """trainDS = NerDataset(df_trainDS)
    devDS = NerDataset(df_devDS)
    testDS = NerDataset(df_testDS)
    """
    trainDS = Ner_Data_Sentence(train_file)
    devDS = Ner_Data_Sentence(dev_file)
    testDS = Ner_Data_Sentence(test_file)

    label_to_ids = trainDS.label_to_ids
    ids_to_label = trainDS.ids_to_label
    ignoreToken='<pad>'
    ignore_token=label_to_ids[ignoreToken]
    print(label_to_ids)
    print(ids_to_label)
    #print(trainDS)
    #print(trainDS[2])
    print(len(trainDS[0]['input_ids']))
    print(len(trainDS[0]['words']))
    print(len(trainDS[0]['labels']))
    print(trainDS[0]['input_ids'])
    print(trainDS[0]['words'])
    print(trainDS[0]['labels'])

    print(len(trainDS[1]['input_ids']))
    print(len(trainDS[1]['words']))
    print(len(trainDS[1]['labels']))
    print(trainDS[1]['input_ids'])
    print(trainDS[1]['words'])
    print(trainDS[1]['labels'])

    #train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    #dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                trainDS,  # The training samples.
                sampler = RandomSampler(trainDS), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                collate_fn=padding
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                devDS, # The validation samples.
                sampler = SequentialSampler(devDS), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                collate_fn=padding
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                testDS, # The validation samples.
                sampler = SequentialSampler(testDS), # Pull out batches sequentially.
                batch_size = batch_size,# Evaluate with this batch size.
                collate_fn=padding
            )        

    print("optimizer ...")
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                  lr = learningRate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = epsilone # args.adam_epsilon  - default is 1e-8.
                )
    
    num_training_steps = num_epochs * len(train_dataloader)
    """lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps
    )"""
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmeup_step, # Default value in run_glue.py
                                            num_training_steps = num_training_steps)
    results=[]
    # Set the seed value all over the place to make this reproducible.
    for seed_val in seed_vals:
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        
        total_t0 = time.time()
        allpreds = []
        alllabels = []
        es=EarlyStopping(patience=es_patience)
        # For each epoch...
        for epoch_i in range(0, num_epochs):
            train_metrics = MetricsTracking("train",ids_to_label,tokenizer)
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
            print('Training ...')
            t0 = time.time()
            total_train_loss = 0
            model.train()
            sqz=1
            for step,batch_item in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                b_input_ids =batch_item['input_ids'].to(device)
                b_input_mask = batch_item['attention_mask'].to(device)
                b_labels = batch_item['labels'].to(device)
                
                model.zero_grad()        
                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss, logits=outputs[:2] 
                #print('train outputs',outputs)
                predictions = logits.argmax(dim= -1) 
                #compute metrics
                #train_metrics.update(predictions, batch_label)
                total_train_loss += loss#.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            
            t0 = time.time()
            model.eval()
            dev_metrics = MetricsTracking('dev',ids_to_label,tokenizer)

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            allpreds = []
            alllabels = []
            last_batch=[]
            allWords,allHeads,allTags,allInputs=[],[],[],[]
            with torch.no_grad(): 
                # Evaluate data for one epoch
                for batch_item in validation_dataloader:
                    last_batch=batch_item
                    b_input_ids = batch_item['input_ids'].to(device)
                    b_input_mask = batch_item['attention_mask'].to(device)
                    b_labels = batch_item['labels'].to(device)
                    b_words=batch_item['words']
                    b_heads=batch_item['attention_mask']
                    b_tags=batch_item['originalLabels']
                    assert(len(b_input_ids)==len(b_labels)==len(b_input_mask)), "the size of the inpits are not the same to feed in the model"
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                    loss, logits=outputs.loss,outputs.logits #[:2]    
                    
                    predictions = logits.argmax(dim= -1)
                    predictions = predictions.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    b_heads = b_heads.detach().cpu().numpy()
                    #print('batch_input ids',b_input_ids)
                    #print('batch_labels',b_labels)
                    #print('batch_predictions',predictions)
                    # Accumulate the validation loss.
                    total_eval_loss += loss#.item()

                    allpreds.extend(predictions)
                    allInputs.extend(b_input_ids)
                    allHeads.extend(b_heads)
                    alllabels.extend(label_ids)
                    allWords.extend(b_words)
                    allTags.extend(b_tags)
                    # Move logits and labels to CPU
                    #logits = logits.detach().cpu().numpy()
                    #label_ids = b_labels.to('cpu').numpy()

                    

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    #total_eval_accuracy += flat_accuracy(logits, label_ids)
                    

                    #alllabels.extend(label_ids.flatten())
                    #allpreds.extend(np.argmax(logits, axis=1).flatten())
                    
            
            dev_metrics.update(allpreds, alllabels,allInputs,allWords,allHeads,allTags,epoch_i,ignore_token=ignore_token)
            #train_results = train_metrics.return_avg_metrics(len(train_dataloader))
            dev_results = dev_metrics.return_avg_metrics(1)#len(validation_dataloader))
            
            #print(f"TRAIN \nMetrics {train_results}\n" ) 
            print(f"VALIDATION \nMetrics{dev_results}\n" )
            
            
            # Report the final accuracy for this validation run.
            avg_val_accuracy = dev_results['acc'] #total_eval_accuracy / len(validation_dataloader)
            print("Validation  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            P,R,F1 = dev_results['precision'],dev_results['recall'],dev_results['f1'] #flat_scores(allpreds, alllabels)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
            
            
            print('Evaluation classification report:')
            print(dev_metrics.getClassificationReport())
            #print(classification_report(alllabels, allpreds, zero_division=0))

            ## early stopping using the loss on the dev set -> break from the epoch loop
            if es.step(avg_val_loss.to('cpu')):
                print(f'BREAK from epoch loop with {avg_val_loss} loss in epoch {epoch_i}')
                
                model=lastModel
                
                break
            else:
                lastModel=model

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Valid_Precision_macro':P,
                    'Valid_Recall_macro':R,
                    'Valid_F1_macro':F1,
                    #'Test. Loss': avg_test_loss,
                    #'Test. Accur.': avg_test_accuracy,
                    'Training Time': training_time,
                    'Valididation Time': validation_time,
                    #'Test Time': test_time
                    'num_patience':es_patience,
                    'seed_val':seed_val,
                    'run':f'{runMessage}#S{seed_val}#P{es_patience}'
                }
            )
            
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))                                                                                        
        saveModel(model=model,tokenizer=tokenizer,stats=training_stats,text=f'{runMessage}#S{seed_val}#P{es_patience}')
        
        
        # ========================================
        #               Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our test set.

        print("")
        print("Running Test...")

        t0 = time.time()
        model.eval()
        testMetric=MetricsTracking('test',ids_to_label,tokenizer)
        # Tracking variables 
        total_test_accuracy = 0
        total_test_loss = 0
        nb_eval_steps = 0
        test_allpreds,test_allHeads,test_allWords,test_allTags,test_allInputs = [],[],[],[],[]
        test_alllabels = []
        # Evaluate data for one epoch
        for batch_item in test_dataloader:
            b_input_ids = batch_item['input_ids'].to(device)
            b_input_mask = batch_item['attention_mask'].to(device)
            b_labels = batch_item['labels'].to(device)
            t_words=batch_item['words']
            t_heads=batch_item['attention_mask']
            t_tags=batch_item['originalLabels']
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits=outputs[:2]    
            # Accumulate the validation loss.
            total_test_loss += loss #.item()

            t_predictions = logits.argmax(dim= -1)
            t_predictions = t_predictions.detach().cpu().numpy()
            t_heads=t_heads.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            test_allpreds.extend(t_predictions)
            test_allInputs.extend(b_input_ids)
            test_allHeads.extend(t_heads)
            test_alllabels.extend(label_ids)
            test_allWords.extend(t_words)
            test_allTags.extend(t_tags)
            # Move logits and labels to CPU
            #logits = logits.detach().cpu().numpy()
            #label_ids = b_labels.to('cpu').numpy()
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            #total_test_accuracy += flat_accuracy(logits, label_ids)
            
            #test_alllabels.extend(label_ids.flatten())
            #test_allpreds.extend(np.argmax(logits, axis=1).flatten())
            

        # Report the final accuracy for this test run.
        testMetric.update(test_allpreds, test_alllabels,test_allInputs,test_allWords,test_allHeads,test_allTags,epoch_i,ignore_token=ignore_token)

        test_Results = testMetric.return_avg_metrics(1)#len(test_dataloader))

        avg_test_accuracy = test_Results['acc']
        print("Test  Accuracy: {0:.2f}".format(avg_test_accuracy))

        P_test,R_test,F1_test= test_Results['precision'],test_Results['recall'],test_Results['f1'] #flat_scores(test_allpreds, test_alllabels)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)
        
        # Measure how long the test run took.
        test_time = format_time(time.time() - t0)
        
        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))  
        results.append(
            {
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Valid_Precision_macro':P,
                'Valid_Recall_macro':R,
                'Valid_F1_macro':F1,
                'Test. Loss': avg_test_loss,
                'Test. Accur.': avg_test_accuracy,
                'Test_Precision_macro':P_test,
                'Test_Recall_macro':R_test,
                'Test_F1_macro':F1_test,
                'Training Time': training_time,
                'Valididation Time': validation_time,
                'Test Time': test_time,
                'num_patience':es_patience,
                'seed_val':seed_val,
                'run':f'{runMessage}#S{seed_val}#P{es_patience}'
            }
        )

        

        print('Test classification report')
        print(testMetric.getClassificationReport())

    
    output_dir = f'model_save/results'
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving results to %s" % output_dir)
    results_df=pd.DataFrame(data=results)
    results_df.to_csv(os.path.join(output_dir, f'result_stats_{runMessage}.csv'))
    print(results_df[['Valid. Accur.','Valid_Precision_macro','Valid_Recall_macro','Valid_F1_macro','Test. Accur.','Test_Precision_macro','Test_Recall_macro','Test_F1_macro']].describe())
    



if __name__=='__main__':
    main()
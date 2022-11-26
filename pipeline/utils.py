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

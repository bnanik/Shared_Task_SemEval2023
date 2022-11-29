#pip install transformers
import pandas as pd
import json,re,os
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler, SequentialSampler
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
from torch import nn

class ElectraClassifier(nn.Module):

    def __init__(self, model_name,dropout=0.5,hidden_layers=50,num_classes=2):

        super(ElectraClassifier, self).__init__()
        #discriminator = ElectraForPreTraining.from_pretrained(model_name)
        self.electra =ElectraModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()
        
        #self.dropout = nn.Dropout(dropout)
        #self.linear = nn.Linear(2, hidden_layers)
        #self.linear2 = nn.Linear(hidden_layers, num_classes)
        #self.relu = nn.ReLU()
        self.sig=nn.Sigmoid()

    def forward(self, input_id, mask,type_ids,lbl):

        #print(f"input_id:{input_id}\nlabel:{lbl}")
        
        logits=[]
        y_hat=[]

        try:
            if input_id is not None:
                if self.training:
                    self.electra.train()
                    output = self.electra(input_ids= input_id, attention_mask=mask,token_type_ids=type_ids) #labels=lbl
                    output_01=output.last_hidden_state[:, 0]
                else:
                    self.electra.eval()
                    #with torch.no_grad:
                    output = self.electra(input_ids= input_id, attention_mask=mask,token_type_ids=type_ids) #labels=lbl
                    output_01=output.last_hidden_state[:, 0]

                
                logits=self.classifier(output_01)
                y_hat=logits.argmax(-1)
                
                #output_0_0=torch.sigmoid(output.last_hidden_state[:, 0])
                #print("electra output",output)
                #output_0 = self.classifier(output_0_0)
                
                #print("before ",output)    
                #print(f"output_0_clssifier:{output_0} ")
                #output_0=output_0.argmax(1)
                #print(f"output_0_argmax:{output_0} ")
                #print(f"output_0_sigmoid :{output} lbl:{lbl}")
                    
            else:
                print(f"NAN input! input_ids: {input_id} \t label: {lbl}")        
        except Exception as e:
            print(f"ERROR! input_ids: {input_id} \t label: {lbl}\n {e}") 
                   
        return logits, y_hat,lbl

        """dropout_output = self.dropout(pooled_output)
        print("dropout_output",dropout_output)

        linear_output = self.linear(dropout_output)
        print("linear_1_output",linear_output)

        linear_2_output = self.linear2(linear_output)
        print("linear_2_output",linear_2_output)



        #final_layer = self.relu(linear_2_output)
        final_layer = self.soft(linear_2_output)
        print("final_layer",final_layer)"""

        #return final_layer
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
class Data_Sentence(Dataset):

    def __init__(self, df,tknrz):
        
        #print(f"values: {self.data['label']}")
        self.sentences=[str(x).strip() for x in df['text'] if len(str(x).strip())>0]
        self.labels=[x for x in df['label']]
        self._labels=self.unique(self.labels)
        #self.labels=torch.from_numpy(np.asarray(self.labels)).type(torch.FloatTensor)
        
        #print("init_sent",self.sentences)
        #print("init_label",self.labels)
        self.tokenizer=tknrz
        
#       print("dataloader initialized")
        
        
    def read_data(self,filename):
        data=pd.read_csv(filename)
        
        self._labels=data['label_sexist'].unique()
        data['label_sexist']=data['label_sexist'].map(self.label_to_ids)
        data=data.filter(items=["text_clean_final","label_sexist"],axis=1)
        data.columns=['text','label']
        #print("data",data)
        return data

    def unique(self,list1):        # initialize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    
    @property  
    def label_to_ids(self):
        return {tag:idx for idx, tag in enumerate(self._labels)}

    
    @property
    def ids_to_label(self):
        return {idx:tag for idx, tag in enumerate(self._labels)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        #print(idx)
        #print(f'sentence: sents: idx: {idx}  {self.sentences[idx]}\nlabels: {self.labels[idx]}')
        sents, lbls = self.sentences[idx], self.labels[idx]

         
        sen_code = self.tokenizer.encode_plus(sents,       #tokenizer.encode_plus(..)
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 128,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            truncation=True,
            #return_tensors = 'pt'
            )


        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] =lbls # torch.as_tensor(lbls)
        item['originalLabels'] = lbls
        item['sentences']=sents

        return item


class MetricsTracking():
  """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """
  def __init__(self,title,ids_ToLabel,run_message):

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
    self.current_pred=[]
    self.current_tags=[]
    self.current_words=[]
    self.runMessage=run_message

  def getClassificationReport(self):
        #tags = list(set(w for w in self.current_tags[1:-1]))
        #print('tags',tags)
        return classification_report(self.total_predictions,self.total_label, zero_division=0)

  def save_results(self,predictions, labels ,inputs,epoch):
    output_dir = f'SEMEVAL/model_save/results'
    f_name=os.path.join(output_dir, f'details_{self.title}_{self.runMessage}.txt')
    with open(f_name, 'a') as fout:
        for idx,item  in enumerate(list(zip(inputs,labels, predictions))):
                token=item[0]
                tag=item[1]
                y_hat=item[2]
                print(f'epoch:{epoch}\n,token:{token} \n,tag: {tag}\n,y_hat: {y_hat}\n')
                fout.write(f"{epoch} {str(token)} {tag} {y_hat} \n")
                
    fout.close()    

  def update(self, predictions, labels ,inputs,words=[],tags=[],epoch=0, ignore_token = -100):
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
    self.tags.extend(tags)
    n_labels=[]
    n_predictions=[]
    n_input_ids=[]
    #print('words',f'{words} tag:{tags}  pred:{predictions} labels:{labels}  heads:{heads}')
    for idx,item  in enumerate(list(zip(words,labels, predictions))):

                token=item[0]
                tag=item[1]
                y_hat=item[2]
                #preds=[self.idsToLabel[x] for x,y in zip(y_hat,hd) if y==1]
                print('input token',f'{token} tag:{tag}  pred:{y_hat}')
                n_labels.append(tag)
                n_predictions.append(y_hat)
                n_input_ids.append(token)
                """try:
                    #assert len(preds)==len(token.split())==len(tag.split())
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
                    continue """          

    
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

    
    #self.save_results(n_predictions,n_labels,n_input_ids,epoch)

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

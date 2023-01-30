#pip install transformers
#!pip install nltk
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
import collections
"""import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer"""
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
#from nltk.corpus import wordnet as wn
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix, classification_report#,ConfusionMatrixDisplay
from sklearn import metrics
import numpy as np
from string import punctuation

import re
#import nltk
import string

import statistics
"""
from nltk.stem.porter import PorterStemmer




nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
"""
class ElectraClassifier(nn.Module):

   

    def __init__(self, model_name,device,dropout=0.5,hidden_layers=50,num_classes=2):

        super(ElectraClassifier, self).__init__()
        self.device=device
        #discriminator = ElectraForPreTraining.from_pretrained(model_name)
        self.electra =ElectraModel.from_pretrained(model_name) # ,from_tf=True
        
        D_in=self.electra.config.hidden_size
        H=20
        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ("lineer1", nn.Linear(D_in, H)),
                    ("relu1", nn.ReLU()),
                    ("dropout", nn.Dropout(0.5)),
                    ("out", nn.Linear(H, num_classes)),
                ]
             )
            
        )


    def forward(self, input_id, mask=None,type_ids=None,lbl=None):

        #print(f"input_id:{input_id}\nlabel:{lbl}")
        
        logits=[]
        y_hat=[]
        input_id.to(self.device)
        if mask is not None:
            mask.to(self.device)
        if type_ids is not None:
            type_ids.to(self.device)
        if lbl is not None:
            lbl.to(self.device)
        try:
            if input_id is not None:
                if self.training:
                    #print("state training in nn.module")
                    self.electra.train()
                    output = self.electra(input_ids= input_id, attention_mask=mask,token_type_ids=type_ids)
                    output_01=output.last_hidden_state[:, 0]
                    logits = self.classifier(output_01)
                    y_hat=logits.argmax(-1)
                else:
                    self.electra.eval()
                    with torch.no_grad():
                        output = self.electra(input_ids= input_id, attention_mask=mask,token_type_ids=type_ids)
                        output_01=output.last_hidden_state[:, 0]
                        logits = self.classifier(output_01)
                        y_hat=logits.argmax(-1)
                    
            else:
                print(f"NAN input! input_ids: {input_id} \t label: {lbl if lbl is not None else 0}")        
        except Exception as e:
            print(f"ERROR! input_ids: {input_id} \t label: {lbl if lbl is not None else 0}\n {e}") 
                   
        return logits,lbl,y_hat
        """if self.training:
            return logits,y_hat, lbl
        else:
            return logits,y_hat"""

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
        
        self._labels=data['sexist'].unique()
        data['sexist']=data['sexist'].map(self.label_to_ids)
        data=data.filter(items=["text","sexist"],axis=1)
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
        item['labels'] =torch.as_tensor(lbls)
        item['originalLabels'] = lbls
        item['sentences']=sents

        return item

class Data_Sentence_Test(Dataset):

    def __init__(self, df,tknrz):
        
        #print(f"values: {self.data['label']}")
        self.sentences=[str(x).strip() for x in df['text'] if len(str(x).strip())>0]
        self.rewire_ids=[str(x).strip() for x in df['rewire_id'] if len(str(x).strip())>0]
        #self.labels=[x for x in df['label']]
        #self._labels=self.unique(self.labels)
        #self.labels=torch.from_numpy(np.asarray(self.labels)).type(torch.FloatTensor)
        
        #print("init_sent",self.sentences)
        #print("init_label",self.labels)
        self.tokenizer=tknrz
        
#       print("dataloader initialized")
        
        
    """def read_data(self,filename):
        data=pd.read_csv(filename)
        
        #self._labels=data['label_sexist'].unique()
        #data['label_sexist']=data['label_sexist'].map(self.label_to_ids)
        data=data.filter(items=["text_clean_final","label_sexist"],axis=1)
        data.columns=['text','label']
        #print("data",data)
        return data"""

    def unique(self,list1):        # initialize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    
    
    
    
    def __len__(self):
        return len(self.rewire_ids)

    def __getitem__(self, idx):
        #print(idx)
        #print(f'sentence: sents: idx: {idx}  {self.sentences[idx]}\nlabels: {self.labels[idx]}')
        sents, rewire_ids = self.sentences[idx], self.rewire_ids[idx]

         
        sen_code = self.tokenizer.encode_plus(sents,       #tokenizer.encode_plus(..)
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 128,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            truncation=True,
            #return_tensors = 'pt'
            )


        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        #item['labels'] =torch.as_tensor(lbls)
        item['rewire_ids'] = rewire_ids
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
    try:
        output_dir = f'model_save/results'
        f_name=os.path.join(output_dir, f'details_{self.title}_{self.runMessage}.txt')
        with open(f_name, 'a') as fout:
            for idx,item  in enumerate(list(zip(inputs,labels, predictions))):
                    token=item[0]
                    tag=item[1]
                    y_hat=item[2]
                    #print(f'epoch:{epoch}\n,token:{token} \n,tag: {tag}\n,y_hat: {y_hat}\n')
                    fout.write(f"{epoch} {str(token)} {tag} {y_hat} \n")
                    
        fout.close()    
    except:
        print('error in saving results')    
  def save_outputs(self,predictions, tags ,words,epoch,runMessage):
    try:
        output_dir = f'model_save/results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        f_name=f'{output_dir}/dtl_elc_{self.title}_{runMessage}.txt'
        with open(f_name, 'w') as fout:
            try:
                fout.write(f"text,actual,pred \n")
                for idx,item  in enumerate(list(zip(words,tags, predictions))):

                    token=item[0]
                    tag=item[1]
                    y_hat=self.idsToLabel[item[2]]
                    fout.write(f"{str(token)},{tag},{y_hat} \n")
                    
            except:
                print(f'ERROR!  input token',f'{token} tag:{tag}  pred:{y_hat}') 
                
        fout.close()    
    except:
        print("An error occured with saving the outputs")    
     
  def update(self, predictions, labels ,inputs,words=[],tags=[],epoch=0, ignore_token = -100):
    '''
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100

    '''
    try:  
        #if torch.is_tensor(predictions):
        predictions=[x for x in predictions]
        
        #if torch.is_tensor(labels):
        labels=[x for x in labels]   
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

        #print(f'pred:{predictions} labels:{labels}')
        for idx,item  in enumerate(list(zip(words,labels, predictions))):

                    token=item[0]
                    tag=item[1]
                    y_hat=item[2]
                    #preds=[self.idsToLabel[x] for x,y in zip(y_hat,hd) if y==1]
                    #print('input token',f'{token} tag:{tag}  pred:{y_hat}')
                    n_labels.append(tag)
                    n_predictions.append(y_hat)
                    n_input_ids.append(token)
        
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
    except Exception as ex:
        print(f"ERROR in metric update. ex: {ex}")    

  
  def return_avg_metrics(self,data_loader_size):
    n = data_loader_size
    metrics = {
        "acc": round(self.total_acc / n ,3), 
        "f1": round(self.total_f1 / n, 3), 
        "precision" : round(self.total_precision / n, 3), 
        "recall": round(self.total_recall / n, 3)
          }
    return metrics   


EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}
chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""
PUNCT_TO_REMOVE = string.punctuation
STOPWORDS =stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]  # set(stopwords.words('english'))
# Chat words removal
chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

# Punctuation removal:
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

#Stop words removal:
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

"""# Stemming
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# Lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])"""

# Emojis Removal
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Emoticons removal
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

# URL and Html tag removal
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    t= url_pattern.sub(r'', text)
    html=re.compile(r'<.*?>') 
    return html.sub(r'',t) #Removing html tags

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
    
#cleaning combinations:
    # 0- lowercase
    # 1- lowercase, punctuation, stopwords
    # 2- lowercase, punctuation, stopwords,lemma,,url/tags
    # 3- lowercase, punctuation, stopwords,lemma,,url/tags,emoji,chatwords
    

# combination 0
def clean_text0(text):
    text = text.lower()
    return text

# combination 1
def clean_text1(text):
    
    text = text.lower()
    text= remove_punctuation(text)
    #text=remove_stopwords(text)
  
    return text

# combination 2
def clean_text2(text):
    # Chat words removal
    chat_words_map_dict = {}
    chat_words_list = []
    for line in chat_words_str.split("\n"):
        if line != "":
            cw = line.split("=")[0]
            cw_expanded = line.split("=")[1]
            chat_words_list.append(cw)
            chat_words_map_dict[cw] = cw_expanded
    chat_words_list = set(chat_words_list)
    text = text.lower()
    text= remove_punctuation(text)
    #text=remove_stopwords(text)
    #text=stem_words(text)
    #text=lemmatize_words(text)
    text= remove_urls(text)
    return text
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

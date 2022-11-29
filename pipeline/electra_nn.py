#!pip install sklearn numpy transformers datasets 
import pandas as pd
import json,re,os
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer,BertForTokenClassification,BertForSequenceClassification,BertTokenizer, BertConfig,AutoTokenizer
from transformers import TrainingArguments, Trainer ,AdamW,get_linear_schedule_with_warmup
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import ElectraConfig, ElectraModel,ElectraTokenizer,ElectraForSequenceClassification
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,precision_recall_fscore_support, f1_score, precision_score,recall_score
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import time
import datetime
import random
from torch import nn
from tqdm import tqdm
from utils import EarlyStopping,MetricsTracking,Data_Sentence,ElectraClassifier




#%% configs

# run #22
runMessage='run#1'
num_epochs = 1
batch_size = 32
learningRate=1e-5
epsilone=1e-8
warmeup_step=0
seed_vals = [42]
es_patience=1

weight_decay=0.01
no_decay = ['bias', 'LayerNorm.weight']
labels__=['not sexist','sexist']
checkpoint_name='my_checkpoint.pt'
global label_to_ids,ids_to_label
label_to_ids = {'not sexist':0,'sexist':1}
ids_to_label = {0:'not sexist',1:'sexist'}
#%% main functionalities
# early stopping in number of epochs


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

def train(model:ElectraClassifier,tokenizer:ElectraTokenizer,train_dataloader,validation_dataloader,device,ids_to_label,seed=42,do_train=False):
    print("optimizer ...")

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                lr = learningRate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = epsilone # args.adam_epsilon  - default is 1e-8.
                )
    
    num_training_steps = num_epochs * len(train_dataloader)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmeup_step, # Default value in run_glue.py
                                            num_training_steps = num_training_steps)                             
    training_stats = []
    PATH=f"model/{seed}_{checkpoint_name}"
    if do_train==False:
        try:
            model.load_state_dict(torch.load(PATH))
        except:
            print(f"The model {PATH} does not exist")  
    else:            
        total_t0 = time.time()
        allpreds = []
        alllabels = []
        model.to(device)
        es=EarlyStopping(patience=es_patience)
        criteion=nn.CrossEntropyLoss(ignore_index=0)
        # For each epoch...
        for epoch_i in range(0, num_epochs):
        
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
                b_token_type_ids=batch_item['token_type_ids'].to(device)
                b_input_mask = batch_item['attention_mask'].to(device)
                b_labels = batch_item['labels'].to(device)
                #print(f"labels:{b_labels}   ")
                model.zero_grad()        
                outputs = model(b_input_ids,b_input_mask,b_token_type_ids,b_labels)
                logits,y_hat,y=outputs[0],outputs[1],outputs[2] 
                logits_t=logits.view(-1,logits.shape[-1])
                y_hat=y_hat.view(-1)
                loss=criteion(logits_t,y)
                print('train loss',loss)
                #print('train lgits',logits_t)
                predictions = logits_t #logits.argmax(dim= -1) 
                #compute metrics
                #train_metrics.update(predictions, batch_label)
                total_train_loss += loss#.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = format_time(time.time() - t0)
            

            torch.save(model.state_dict(), PATH)
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
            dev_metrics = MetricsTracking('dev',ids_to_label,run_message=runMessage)

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
                    b_token_type_ids=batch_item['token_type_ids'].to(device)
                    b_input_mask = batch_item['attention_mask'].to(device)
                    
                    b_labels = batch_item['labels'].to(device)
                    b_sentences=batch_item['sentences']
                    print(f"original labels{batch_item['labels']}\nb_labels:{b_labels}")
                    #assert(len(b_input_ids)==len(b_labels)==len(b_input_mask)), "the size of the inpits are not the same to feed in the model"
                    outputs = model(b_input_ids, 
                                    b_input_mask,
                                    b_token_type_ids,
                                    b_labels)
                    logits,y_hat,y=outputs[0],outputs[1] ,outputs[2]  
                    #print(f"outputs {outputs}\nloss:{loss}\nlogits:{logits}")
                    #logits_t = logits#.argmax(dim= -1)
                    
                    #predictions = logits_t.detach().cpu()
                    #label_ids = b_labels.detach().cpu()
                    #print('batch_input ids',b_input_ids)
                    #print('batch_labels',b_labels)
                    #print('batch_predictisssssons',predictions)
                    # Accumulate the validation loss.
                    #total_eval_loss += loss#.item()
                    #print(f"labels:{label_ids}\nlogits:{logits}\nlogits_t: {logits_t}\npredictions: {predictions}")
                    allpreds.extend(y_hat)#.numpy().tolist())
                    allInputs.extend(b_input_ids)
                    alllabels.extend(y)#.numpy().tolist())
                    allWords.extend(b_sentences)
                    #allTags.extend(b_tags)
                    # Move logits and labels to CPU
                    #logits = logits.detach().cpu().numpy()
                    #label_ids = b_labels.to('cpu').numpy()

                    

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    #total_eval_accuracy += flat_accuracy(logits, label_ids)
                    

                    #alllabels.extend(label_ids.flatten())
                    #allpreds.extend(np.argmax(logits, axis=1).flatten())
                    
            
            dev_metrics.update(allpreds, alllabels,allInputs,allWords,allHeads,allTags,epoch_i)
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
                    'seed_val':seed,
                    'run':f'{runMessage}#S{seed}#P{es_patience}'
                }
            )
            
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))                                                                                        
        saveModel(model=model,tokenizer=tokenizer,stats=training_stats,text=f'{runMessage}#S{seed}#P{es_patience}')
    
    return model
    
    
def evaluate(model:ElectraModel,test_dataloader,ids_to_label,device,seed=42):
    # ========================================
    #               Test
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our test set.
    results=[]
    print("")
    print("Running Test...")

    t0 = time.time()
    model.eval()
    testMetric=MetricsTracking('test',ids_to_label,run_message=runMessage)
    # Tracking variables 
    total_test_accuracy = 0
    total_test_loss = 0
    nb_eval_steps = 0
    test_allpreds,test_allSentences,test_allInputs = [],[],[]
    test_alllabels = []
    # Evaluate data for one epoch
    for batch_item in test_dataloader:
        b_input_ids = batch_item['input_ids'].to(device)
        b_token_type_ids=batch_item['token_type_ids'].to(device)
        b_input_mask = batch_item['attention_mask'].to(device)
        b_labels = batch_item['labels'].to(device)
        t_sentences=batch_item['sentences']
        t_heads=batch_item['attention_mask']
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                b_input_mask,
                                b_token_type_ids,
                                b_labels)
        logits,y_hat,y=outputs[0],outputs[1],outputs[2]    
        # Accumulate the validation loss.
        loss=0
        total_test_loss += loss #.item()

        #t_predictions = logits#.argmax(dim= -1)
        t_predictions = y_hat#.numpy().tolist()
        label_ids = y#.numpy().tolist()
        print(f"labels:{b_labels}\npredictions:{t_predictions}")
        test_allpreds.extend(t_predictions)
        test_allInputs.extend(b_input_ids)
        test_alllabels.extend(label_ids)
        test_allSentences.extend(t_sentences)
        # Move logits and labels to CPU

    # Report the final accuracy for this test run.
    testMetric.update(test_allpreds, test_alllabels,test_allInputs,test_allSentences,ignore_token=-100)

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
            'Test. Loss': avg_test_loss,
            'Test. Accur.': avg_test_accuracy,
            'Test_Precision_macro':P_test,
            'Test_Recall_macro':R_test,
            'Test_F1_macro':F1_test,
            'Test Time': test_time,
            'num_patience':es_patience,
            'seed_val':seed,
            'run':f'{runMessage}#S{seed}#P{es_patience}'
        }
    )

    

    print('Test classification report')
    print(testMetric.getClassificationReport())
def read_data(filename):
        data=pd.read_csv(filename)
        
        _labels=data['label_sexist'].unique()
        label_to_ids={tag:idx for idx, tag in enumerate(_labels)}
        ids_to_label={idx:tag for idx, tag in enumerate(_labels)}
        data['label_sexist']=data['label_sexist'].map(label_to_ids)
        data=data.filter(items=["text_clean_final","label_sexist"],axis=1)
        data.columns=['text','label']
        #print("data",data)
        return data,ids_to_label,label_to_ids
def main():
    max_len=0
    is_train=False
    # cuda avaliability?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.manual_seed(seed_val)
    #model definitison
    model_id=  "google/electra-base-discriminator" #"bhadresh-savani/electra-base-emotion" #  "google/electra-small-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_id)
    electra_config = ElectraConfig.from_pretrained(model_id)

    #model = ElectraForSequenceClassification.from_pretrained(model_id)
    
    
    #preparing dataframes
    local="./data/"
    #slocal=""
    train_file=f'{local}clean_data_v2.csv'
    test_file=f'{local}dev_task_a_entries.csv'
  
    df,ids_to_label,label_to_ids=read_data(train_file)

    #print(len(DS[1]['input_ids']))
    #print(len(DS[1]['sentences']))
    #print(len(DS[1]['labels']))
    #print(DS[1]['input_ids'])
    #print(DS[1]['sentences'])
    #print(DS[1]['originalLabels'])
    #print(DS[1]['labels'])

    results=[]
    # Set the seed value all over the place to make this reproducible.
    for seed_val in seed_vals:
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        trainDF, devDF, testDF = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])
        trainDS, devDS, testDS=Data_Sentence(trainDF,tokenizer), Data_Sentence(devDF,tokenizer),Data_Sentence(testDF,tokenizer)
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    trainDS,  # The training samples.
                    sampler = RandomSampler(trainDS), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                    #,collate_fn=padding
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    devDS, # The validation samples.
                    sampler = SequentialSampler(devDS), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                    #,collate_fn=padding
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        test_dataloader = DataLoader(
                    testDS, # The validation samples.
                    sampler = SequentialSampler(testDS), # Pull out batches sequentially.
                    batch_size = batch_size# Evaluate with this batch size.
                    #,collate_fn=padding
                )        

        model=ElectraClassifier(model_name=model_id)
        model=train(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,validation_dataloader=validation_dataloader,
            device=device,ids_to_label=ids_to_label,seed=seed_val,do_train=is_train)
        
        evaluate(model=model,test_dataloader=test_dataloader,device=device,ids_to_label=ids_to_label,seed=seed_val)

    
    """output_dir = f'SEMEVAL/model_save/results'
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving results to %s" % output_dir)
    results_df=pd.DataFrame(data=results)
    results_df.to_csv(os.path.join(output_dir, f'result_stats_{runMessage}.csv'))
    print(results_df[['Valid. Accur.','Valid_Precision_macro','Valid_Recall_macro','Valid_F1_macro','Test. Accur.','Test_Precision_macro','Test_Recall_macro','Test_F1_macro']].describe())
    """



if __name__=='__main__':
    main()
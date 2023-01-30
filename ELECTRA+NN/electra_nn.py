#!py -m pip install sklearn numpy transformers datasets pandas 
import pandas as pd
import json,re,os
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler, SequentialSampler
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
from utils import *
import torch.nn.functional as F
import argparse
import pickle 
#%% configs
# run #15
runMessage='run#15' 
num_epochs = 20
batch_size = 32
learningRate=2e-5
epsilone=1e-8
warmeup_step=0
seed_vals = [42]
es_patience=20


weight_decay=0.01
no_decay = ['bias', 'LayerNorm.weight']
labels__=['not sexist','sexist']
checkpoint_name='my_checkpoint.pt'
global label_to_ids,ids_to_label
label_to_ids = {'not sexist':0,'sexist':1}
ids_to_label = {0:'not sexist',1:'sexist'}
#%% main functionalities


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



def load_model(model_id,device,text='run'):
  
    output_dir = f'{text}'
    nn_path = os.path.join(output_dir,'nn.pickle')
    
    transformer = ElectraModel.from_pretrained(output_dir)
    model=ElectraClassifier(model_name=model_id,device=device)
    model.electra=transformer
    
    nn = pickle.load(open(nn_path, 'rb'))
    model.classifier.load_state_dict(nn)

    tokenizer=ElectraTokenizer.from_pretrained(output_dir)
    
    return model,tokenizer
   
def saveModel(model:ElectraClassifier,tokenizer:ElectraTokenizer,stats,text='run'):
    ts=time.time()
    output_dir = f'model_save/{text}'
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    
    transformer = model.electra
    transformer.save_pretrained(output_dir)
    
    nn_path = os.path.join(output_dir,'nn.pickle')
    torch.save(model.classifier.state_dict(), nn_path)
    
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #model_to_save.savsave_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    training_stats_df=pd.DataFrame(data=stats)
    training_stats_df.to_csv(os.path.join(output_dir, f'training_stats_{runMessage}.csv'))


def train(model:ElectraClassifier,tokenizer:ElectraTokenizer,train_dataloader,validation_dataloader,test_dataloader1,test_dataloader2,device,ids_to_label,seed=42,do_train=False,runMessage='def'):
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
    loss_fn = nn.CrossEntropyLoss()
    local="" #"./pipeline/"
    PATH=f"{local}model/{seed}_{checkpoint_name}"
    if do_train=='False':
        try:
            model.load_state_dict(torch.load(PATH))
            print(f"Load model from {PATH} succeeded.")
        except:
            print(f"The model {PATH} does not exist")  
    else:            
        total_t0 = time.time()
        allpreds = []
        alllabels = []
        model.to(device)
        es=EarlyStopping(patience=es_patience)

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
                optimizer.zero_grad()  
                #print(f'Devices: model {next(model.parameters()).device} b_input_ids: {b_input_ids.device}   b_input_mask: {b_input_mask.device}     b_token_type_ids: {b_token_type_ids.device}   b_labels:{b_labels.device}')
                outputs = model(b_input_ids,b_input_mask,b_token_type_ids,b_labels)
                #print("outputs",outputs)
                logits,y,y_hat=outputs[0],outputs[1],outputs[2]
                loss = loss_fn(logits, b_labels) 

                #print('train loss',loss.item())
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = format_time(time.time() - t0)
            

            #torch.save(model.state_dict(), PATH)
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
            total_eval_accuracy = []
            total_eval_loss = []
            nb_eval_steps = 0
            allpreds = []
            alllabels = []
            last_batch=[]
            best_preds=[]
            best_words=[]
            best_tags=[]
            best_input_ids=[]
            best_acc=0
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
                    #print(f"original labels{batch_item['labels']}\nb_labels:{b_labels}")
                    #assert(len(b_input_ids)==len(b_labels)==len(b_input_mask)), "the size of the inpits are not the same to feed in the model"
                    outputs = model(b_input_ids, 
                                    b_input_mask,
                                    b_token_type_ids,
                                    b_labels
                                    )
                    logits,y,y_hat=outputs[0],outputs[1],outputs[2]  
                    # Compute loss
                    loss = loss_fn(logits,b_labels)
                    total_eval_loss.append(loss.item())

                    # Get the predictions
                    preds = y_hat.cpu().numpy().tolist()  #torch.argmax(logits, dim=1).flatten()
                    label_ids=y.cpu().numpy().tolist()
                    # Calculate the accuracy rate
                    accuracy  = sum(p == t for p, t in zip(preds, label_ids))/len(preds) * 100
                    
                    total_eval_accuracy.append(accuracy)
                    #print(f"labels:{label_ids}\nlogits:{logits}\nlogits_t: {logits_t}\npredictions: {predictions}")
                    allpreds.extend(preds)#.numpy().tolist())
                    allInputs.extend(b_input_ids)
                    alllabels.extend(label_ids)#.numpy().tolist())
                    allWords.extend(b_sentences)
            
            val_loss = np.mean(total_eval_loss)
            
            val_accuracy = np.mean(total_eval_accuracy)
            print(f"validation loss_mean: {val_loss} \t accuracy-mean: {val_accuracy}")
            dev_metrics.update(allpreds, alllabels,allInputs,allWords,allHeads,allTags,epoch_i)
            #train_results = train_metrics.return_avg_metrics(len(train_dataloader))
            dev_results = dev_metrics.return_avg_metrics(1)#len(validation_dataloader))
            
            #print(f"TRAIN \nMetrics {train_results}\n" ) 
            print(f"VALIDATION \nMetrics{dev_results}\n" )
            
            
            # Report the final accuracy for this validation run.
            avg_val_accuracy = dev_results['acc'] #total_eval_accuracy / len(validation_dataloader)
            print("Validation  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = val_loss#total_eval_loss / len(validation_dataloader)
            if val_accuracy > best_acc:
                best_tags=alllabels
                best_preds=allpreds
                best_words=allWords
                best_acc=val_accuracy
                best_model=model
            P,R,F1 = dev_results['precision'],dev_results['recall'],dev_results['f1'] #flat_scores(allpreds, alllabels)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
            
            if epoch_i>0 and epoch_i % 19==0:
                d=epoch_i % 19
                evaluateOnUnseenData(model=model,test_dataloader=test_dataloader1,ids_to_label=ids_to_label,device=device,seed=seed,run=f'i_t1_e{epoch_i}_{runMessage}')

                evaluateOnUnseenData(model=model,test_dataloader=test_dataloader2,ids_to_label=ids_to_label,device=device,seed=seed,run=f'i_t2_e{epoch_i}_{runMessage}')

            ## early stopping using the loss on the dev set -> break from the epoch loop
            if es.step(avg_val_loss):
                print(f'BREAK from epoch loop with {avg_val_loss} loss in epoch {epoch_i}')
                
                model=best_model
                print(f'Seed {seed} - Evaluation classification report: epoch: {epoch_i} ')
                dev_metrics.save_outputs(best_preds,best_tags,best_words,epoch_i,f'fl_out_{runMessage}#S{seed}#P{es_patience}#E{epoch_i}')
                print(dev_metrics.getClassificationReport())

                
                break
            
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

        model=best_model
        print(f'Seed {seed} - Evaluation classification report ')
        dev_metrics.save_outputs(best_preds,best_tags,best_words,epoch_i,f'fl_out_{runMessage}#S{seed}')
        print(dev_metrics.getClassificationReport())
        
        #print(f'{runMessage} - validation classification report')
        #print(dev_metrics.getClassificationReport())

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))                                                                                        
        #saveModel(model=model,tokenizer=tokenizer,stats=training_stats,text=f'{runMessage}#S{seed}#P{es_patience}')
    
    return model
    
def evaluate(model:ElectraModel,test_dataloader,ids_to_label,device,seed=42,run='def'):
    loss_fn = nn.CrossEntropyLoss()

    # ========================================
    #               Test
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our test set.
    results=[]
    print("")
    print("Running internal test...")

    t0 = time.time()
    model.eval()
    testMetric=MetricsTracking('test',ids_to_label,run_message=runMessage)
    # Tracking variables 
    total_test_accuracy = []
    total_test_loss = []
    nb_eval_steps = 0
    test_allpreds,test_allSentences,test_allInputs = [],[],[]
    test_alllabels,all_logits = [],[]
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
                                b_labels
                                )
        logits,y,y_hat=outputs[0],outputs[1] ,outputs[2] 
        # Compute loss
        #logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        #y = y.view(-1)  # (N*T,)
        # Get the predictions
        preds = y_hat.cpu().numpy().tolist()  #torch.argmax(logits, dim=1).flatten()
        label_ids=y.cpu().numpy().tolist()
        # Calculate the accuracy rate
        accuracy  = sum(p == t for p, t in zip(preds, label_ids))/len(preds) * 100
        total_test_accuracy.append(accuracy)
      

        # Get predictions from the probabilities
        #threshold = 0.9
        #preds = np.where(probs[:, 1] > threshold, 1, 0)

        t_predictions = preds
        #print(f"labels:{b_labels}\npredictions:{t_predictions}")
        test_allpreds.extend(t_predictions)
        test_allInputs.extend(b_input_ids)
        test_alllabels.extend(label_ids)
        test_allSentences.extend(t_sentences)
        # Move logits and labels to CPU

    # Report the final accuracy for this test run.
    testMetric.update(test_allpreds, test_alllabels,test_allInputs,test_allSentences,ignore_token=-100)

    test_Results = testMetric.return_avg_metrics(1)#len(test_dataloader))

    avg_test_accuracy = np.mean(total_test_accuracy)#test_Results['acc']
    print("Test  Accuracy: {0:.2f}".format(avg_test_accuracy))
    testMetric.save_outputs(test_allpreds,test_alllabels,test_allSentences,0,f'fl_test_out_{runMessage}#S{seed}')
    P_test,R_test,F1_test= test_Results['precision'],test_Results['recall'],test_Results['f1'] #flat_scores(test_allpreds, test_alllabels)

    # Measure how long the test run took.
    test_time = format_time(time.time() - t0)
    
    print("  internal test took: {:}".format(test_time))  
    results.append(
        {
            #'Test. Loss': avg_test_loss,
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

    

    print('{run}- internal test classification report')
    print(testMetric.getClassificationReport())

def evaluateOnUnseenData(model:ElectraModel,test_dataloader,ids_to_label,device,seed=42,run='def'):
    # ========================================
    #               Testing on Unseen Data
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our test set.
    results=[]
    print("")
    print("Running Testing on unseen data...")

    t0 = time.time()
    model.eval()
    # Tracking variables 
    test_allpreds,test_allSentences,all_rewire_ids,test_allInputs = [],[],[],[]
    test_alllabels,all_logits = [],[]
    # Evaluate data for one epoch
    for batch_item in test_dataloader:
        b_input_ids = batch_item['input_ids'].to(device)
        b_token_type_ids=batch_item['token_type_ids'].to(device)
        b_input_mask = batch_item['attention_mask'].to(device)
        t_sentences=batch_item['sentences']
        t_rewire_ids=batch_item['rewire_ids']
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                b_input_mask,
                                b_token_type_ids)
        #print("output",outputs)
        logits,_,y_hat=outputs[0],outputs[1],outputs[2]
        # Get the predictions
        preds = y_hat.cpu().numpy().tolist()  #torch.argmax(logits, dim=1).flatten()
        t_predictions=preds
        #print(f"predictions:{t_predictions}")
        test_allpreds.extend(t_predictions)
        test_allInputs.extend(b_input_ids)
        test_allSentences.extend(t_sentences)
        all_rewire_ids.extend(t_rewire_ids)
    
    
    # Measure how long the test run took.
    test_time = format_time(time.time() - t0)
    
    print("  Unseen Test took: {:}".format(test_time)) 
    try: 
        print("showing/saving unseen data predictions")
        output_dir = f'model_save/results'
        f_name=os.path.join(output_dir, f'details_semeval_unseen_{run}.csv')
        with open(f_name, 'w') as fout:
            fout.write(f"rewire_id,label_pred\n")
            for s,r,p in zip(test_allSentences,all_rewire_ids,test_allpreds):
                #print(f"{s}  {r}  {ids_to_label[p]} \n")
                fout.write(f"{r},{ids_to_label[p]}\n")
            fout.close() 
            
    except Exception as ex:
        print(f"error in showing/saving unseen predictions: {ex}")
    results.append(
        {
            'Test Time': test_time,
            'num_patience':es_patience,
            'seed_val':seed,
            'run':f'{runMessage}#S{seed}#P{es_patience}'
        }
    )


def read_data(filename, type="train"):
        data=pd.read_csv(filename)
        #test on lower nuber of records
        if type=="train":
            #data["text"] = data["text"].apply(lambda x: clean_text1(x))
            _labels=data['sexist'].unique()
            #label_to_ids={tag:idx for idx, tag in enumerate(_labels)}
            #ids_to_label={idx:tag for idx, tag in enumerate(_labels)}
            data['sexist']=data['sexist'].map(label_to_ids)
            
            data=data.filter(items=["text","sexist"],axis=1)
            data.columns=['text','label']
            #print("data",data)
            return data, ids_to_label,label_to_ids
        else:
            data["text"] = data["text"].apply(lambda x: clean_text1(x))
            data=data.filter(items=["rewire_id","text"],axis=1)
            data.columns=['rewire_id','text']
        return data

def main(loc,checkpoint,doTrain,doTestOnUnseenData,train_file,validationfile,test_file,test_file2,run,electra_version):
    max_len=0
    is_train=True
    #doTrain=True
    #doTestOnUnseenData=True
    #checkpoint='None' #'D:\\model_save\\ELECTRA-PT\\output_electra\\latest\\'
    #loc=True
    # cuda avaliability?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.manual_seed(seed_val)
    #model definitison
    model_id2=  f"google/electra-{electra_version}-discriminator" #"bhadresh-savani/electra-base-emotion" #  "google/electra-small-discriminator"
    #model_id2="model_save/run#5#S42#P3/"
    if checkpoint!= 'None':
        model_id=checkpoint
        print(f"running from checkpoint {checkpoint}")
        
    else:
        model_id=model_id2
        print("running from hugging face model")
    
    
    # to load electra original model uncomment this
    #model_id=model_id2
    electra_config = ElectraConfig.from_pretrained(model_id)
    tokenizer = ElectraTokenizer.from_pretrained(model_id)
    model=ElectraClassifier(model_name=model_id,device=device,num_classes=2)

    
    #preparing dataframes
    #local="./data/"
    if loc=="True" :
        local="./data/"
    else:    
        local=""
    
    
    train_file=f'{local}{train_file}'
    
    validation_file=f'{local}{validationfile}'
    test_file=f'{local}{test_file}'

    test_file2=f'{local}{test_file2}'
  
    df,i,j=read_data(train_file)
    testDF,i,j=read_data(validation_file)
    unseen_testDF=read_data(test_file,type="test")
    unseen_testDF2=read_data(test_file2,type="test")
    
    print('ids_to_label',ids_to_label)
    print('label_to_ids',label_to_ids)

    results=[]
    # Set the seed value all over the place to make this reproducible.
    for seed_val in seed_vals:
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        trainDF, devDF = np.split(df.sample(frac=1, random_state=seed_val), [int(.875*len(df))])
        
        """grouped_df = df.groupby('label')
        arr_list = [np.split(g, [int(.875 * len(g))]) for i, g in grouped_df]   # 70/10/20 split

        trainDF = pd.concat([t[0] for t in arr_list])
        devDF = pd.concat([t[1] for t in arr_list])
        #testDF =  pd.concat([v[2] for v in arr_list])"""

        trainDS, devDS, testDS,unseen_testDS=Data_Sentence(trainDF,tokenizer), Data_Sentence(devDF,tokenizer),Data_Sentence( testDF,tokenizer),Data_Sentence_Test(unseen_testDF,tokenizer)
        unseen_testDS2=Data_Sentence_Test(unseen_testDF2,tokenizer)
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
        unseen_test_dataloader = DataLoader(
                    unseen_testDS, # The validation samples for unseen data.
                    sampler = SequentialSampler(unseen_testDS), # Pull out batches sequentially.
                    batch_size = batch_size# Evaluate with this batch size.
                    #,collate_fn=padding
                )  
        unseen_test_dataloader2 = DataLoader(
                    unseen_testDS2, # The validation samples for unseen data.
                    sampler = SequentialSampler(unseen_testDS2), # Pull out batches sequentially.
                    batch_size = batch_size# Evaluate with this batch size.
                    #,collate_fn=padding
                )               

        
        print('doTrain= ',doTrain)
        if doTrain=='True':
            print(f'start training with model {model_id}')
            model=train(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,validation_dataloader=validation_dataloader,
                 test_dataloader1=unseen_test_dataloader,test_dataloader2=unseen_test_dataloader2,device=device,ids_to_label=ids_to_label,seed=seed_val,do_train=bool(doTrain),runMessage=run)
            
            print("evaluating on the internal validation data split from train data ...")
            evaluate(model=model,test_dataloader=test_dataloader,device=device,ids_to_label=ids_to_label,seed=seed_val,run=run)     
        
        if doTestOnUnseenData:
            print(f"testing on useen data: {test_file}...")
            evaluateOnUnseenData(model=model,test_dataloader=unseen_test_dataloader,device=device,ids_to_label=ids_to_label,seed=seed_val,run=f'f_t1_{run}')

            print(f"testing on useen data: {test_file2}...")
            evaluateOnUnseenData(model=model,test_dataloader=unseen_test_dataloader2,device=device,ids_to_label=ids_to_label,seed=seed_val,run=f'f_t2_{run}')
        
            


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ELECTRA finetuning',
                    description = 'finetuning electra',
                    epilog = '')
    
    parser.add_argument('--local', help='the main training dataset file name')
    parser.add_argument('--checkpoint', help='the latest checkpoint')
    parser.add_argument('--doTrain', help='doing the train or not')
    parser.add_argument('--doTestOnUnseenData', help='test on unseen data')
    parser.add_argument('--trainfile', help='train filename')
    parser.add_argument('--validationfile', help='validation filename')
    parser.add_argument('--testfile', help='test1 unseen filename')
    parser.add_argument('--testfile2', help='test2 unseen filename')
    parser.add_argument('--run', help='run message')
    parser.add_argument('--electraversion', help='electra version')

    args = parser.parse_args()  
    print("args",args)

    main(args.local,args.checkpoint,args.doTrain,args.doTestOnUnseenData,args.trainfile,args.validationfile,args.testfile,args.testfile2,args.run,args.electraversion)
  
This repo provides the code and data for our system to address the shared task SemEval 2023, task 10: Explainable Detection of Online Sexism (EDOS).  To address sub-task A,
We developed six models in three different paradigms, namely **further pretraining**, **typical finetuning**, and **prompt-based learning**.

1. **Further Pre-training:** <br />
The whole codes relted to pretrining is vailable in ELECTRA-Pretraining folder. The code is from the official ELECTRA github repo (https://github.com/google-research/electra). To run the pretraining process:  
    1.1 build the dataset  
      > python build_pretraining_dataset.py   --corpus-dir ./corpus  --vocab-file vocab.txt  --output-dir pretrain_tfrecords  --max-seq-length 128   --blanks-separate-docs False   --no-lower-case   --num-processes 10  
      * The corpus is needed to be in the corpus folder with the name of "train_data.txt"  
    
    1.2 run the pretraining based o the parameters json file  
      > python run_pretraining.py --data-dir .  --model-name electra_small --hparams "hparams-small.json"
    
    1.3 convert the tensorflow model to pytorch one:  
            > python convert_electra_original_tf_checkpoint_to_pytorch.py \  
                --tf_checkpoint_path=output_electra_small88000 \  
                --config_file=output_electra_small88000/config.json \  
                --pytorch_dump_path=output_electra_small88000/pytorch/pytorch_model.bin \  
                --discriminator_or_generator=discriminator   
            
 The file hparms.json contains the output directory, number of steps, type of model and other parameters.  
 
2. **Typical Finetuning:** <br />
The main python file is electra_nn for this purpose. It calles other dependecies and related files.
To run the code:
    > python electra_nn.py --local False --checkpoint None --doTrain True --doTestOnUnseenData True --trainfile 'train_EDOS_80.csv' --validationfile 'validation_EDOS_20.csv' --testfile 'dev_task_a_entries.csv' --testfile2 'test_task_a_entries.csv' --run '54_test_HF_large' --electraversion large

* If the local switch set, all data paths will join with a prdefiene "./data" directory. Otherwise the data will be searchd in the root directory. 
 
All the results for the electra+NN model is available in the 'ELECTRA+NN/results' folder.

3. **Prompt-based Learning:** <br /> 
In this approach, we employ T5 PLM in the prompting paradigm. We implement our model in [OpenPrompt](https://github.com/thunlp/OpenPrompt) framework. It supports loading transformer-based models directly from the huggingface. <br/>
Two different experiments have been conducted in this approach: <br />
* Training over EDOS data
  - To repeat this experiment try to run the [Prompt_based_classifier_EDOS_Data.ipynb](https://github.com/bnanik/Shared_Task_SemEval2023/blob/main/Prompt_based_classifier_EDOS_Data.ipynb) file. All the required datasets are available in the data folder. 

* Training over Augmented, Combined, Preprocessed, and Oversampled datasets (The datasets are explained in detail in the paper)
  - To repeat these set of experiments try to run the [prompt_based_classifier_Data_manipulation.ipynb](https://github.com/bnanik/Shared_Task_SemEval2023/blob/main/prompt_based_classifier_Data_manipulation.ipynb) file. All the required datasets are available in the data folder. This file can be run using one of the datasets at a time, namely augmented, combined, preprocessed, and oversampled datasets.

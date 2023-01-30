This repo provides the code and data for our system to address the shared task SemEval 2023, task 10: Explainable Detection of Online Sexism (EDOS).  To address sub-task A,
We developed six models in three different paradigms, namely **further pretraining**, **typical finetuning**, and **prompt-based learning**.

1. **Further Pre-training:** <br />


2. **Typical Finetuning:** <br />


3. **Prompt-based Learning:** <br /> 
In this approach, we employ T5 PLM in the prompting paradigm. We implement our model in [OpenPrompt](https://github.com/thunlp/OpenPrompt) framework. It supports loading transformer-based models directly from the huggingface. <br/>
Two different experiments have been conducted in this approach: <br />
* Training over EDOS data
  - To repeat this experiment try to run the [Prompt_based_classifier_EDOS_Data.ipynb](https://github.com/bnanik/Shared_Task_SemEval2023/blob/main/Prompt_based_classifier_EDOS_Data.ipynb) file. All the required datasets are available in the data folder. 

* Training over Augmented, Combined, Preprocessed, and Oversampled datasets (The datasets are explained in detail in the paper)
  - To repeat these set of experiments try to run the [prompt_based_classifier_Data_manipulation.ipynb](https://github.com/bnanik/Shared_Task_SemEval2023/blob/main/prompt_based_classifier_Data_manipulation.ipynb) file. All the required datasets are available in the data folder. This file can be run using one of the datasets at a time, namely augmented, combined, preprocessed, and oversampled datasets.

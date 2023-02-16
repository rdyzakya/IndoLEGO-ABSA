from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from model import ABSAGenerativeModelWrapper
from datasets import Dataset
from typing import Dict

class ABSAGenerativeTrainer:
    def __init__(self,absa_model_and_tokenizer:ABSAGenerativeModelWrapper):
        """"
        ### DESC
            ABSAGenerativeTrainer constructor.
        ### PARAMS
        * absa_model_and_tokenizer: ABSAGenerativeModelWrapper instance.
        """
        self.model_and_tokenizer = absa_model_and_tokenizer

    def compile_train_args(self,train_args_dict:Dict):
        """
        ### DESC
            Method to load training arguments.
        ### PARAMS
        * train_args_dict: Training arguments (dictionary).
        """
        self.training_args = Seq2SeqTrainingArguments(**train_args_dict) if self.model_and_tokenizer.model_type == "seq2seq" else TrainingArguments(**train_args_dict)
    
    def prepare_data(self,train_dataset:Dataset,eval_dataset:Dataset,test_dataset:Dataset):
        tokenizer = self.model_and_tokenizer.tokenizer

        # Prepare data collator
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if self.model_and_tokenizer.model_type == "seq2seq" else DataCollatorForLanguageModeling(tokenizer=tokenizer)
        
        # Encode the input and output
        pass
    def train(self):
        pass
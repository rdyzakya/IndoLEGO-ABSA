from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import EvalPrediction
from transformers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from model import ABSAGenerativeModelWrapper
from evaluation import recall, precision, f1_score
from datasets import Dataset
from typing import Dict

import numpy as np

class ABSAGenerativeTrainer:
    def __init__(self,absa_model_and_tokenizer:ABSAGenerativeModelWrapper):
        """"
        ### DESC
            ABSAGenerativeTrainer constructor.
        ### PARAMS
        * absa_model_and_tokenizer: ABSAGenerativeModelWrapper instance.
        """
        self.model_and_tokenizer = absa_model_and_tokenizer
    
    def prepare_data(self,train_dataset:Dataset,eval_dataset:Dataset|None=None,test_dataset:Dataset|None=None,**encoding_args):
        """
        ### DESC
            Method for preparing data (data collator and tokenize the dataset).
        ### PARAMS
        * train_dataset: Training dataset.
        * eval_dataset: Eval dataset.
        * test_dataset: Test dataset.
        * encoding_args: Encoding arguments (HF Tokenizer arguments).
        """
        tokenizer = self.model_and_tokenizer.tokenizer
        model_type = self.model_and_tokenizer.model_type

        # Prepare data collator
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if self.model_and_tokenizer.model_type == "seq2seq" else DataCollatorForLanguageModeling(tokenizer=tokenizer)
        
        # Encode the input and output
        if model_type == "seq2seq":
            self.tokenized_train = tokenizer(train_dataset["input"], text_target=train_dataset["output"], **encoding_args)
            if eval_dataset != None:
                self.tokenized_eval = tokenizer(eval_dataset["input"], text_target=eval_dataset["output"], **encoding_args)
            if test_dataset != None:
                self.tokenized_test = tokenizer(test_dataset["input"], text_target=test_dataset["output"], **encoding_args)
        else: # "causal_lm"
            causal_lm_train_input = [train_dataset["input"][i] + ' ' + train_dataset["output"][i] for i in range(len(train_dataset))]
            self.tokenized_train = tokenizer(causal_lm_train_input, **encoding_args)
            if eval_dataset != None:
                causal_lm_eval_input = [eval_dataset["input"][i] + ' ' + eval_dataset["output"][i] for i in range(len(train_dataset))]
                self.tokenized_eval = tokenizer(causal_lm_eval_input, **encoding_args)
            if test_dataset != None:
                causal_lm_test_input = [test_dataset["input"][i] + ' ' + test_dataset["output"][i] for i in range(len(train_dataset))]
                self.tokenized_test = tokenizer(causal_lm_test_input, **encoding_args)
    
    def compute_metrics(self,eval_preds:EvalPrediction):
        preds, labels = eval_preds

        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = np.argmax(preds,axis=-1) if len(preds.shape) == 3 else preds # in case not predict with generate

        pass
        
    def compile_train_args(self,train_args_dict:Dict):
        """
        ### DESC
            Method to load training arguments.
        ### PARAMS
        * train_args_dict: Training arguments (dictionary).
        """
        self.training_args = Seq2SeqTrainingArguments(**train_args_dict) if self.model_and_tokenizer.model_type == "seq2seq" else TrainingArguments(**train_args_dict)

    def prepare_trainer(self):
        """
        ### DESC
            Method for preparing trainer.
        """
        trainer_args = {
            "model" : self.model_and_tokenizer.model,
            "args" : self.training_args,
            "tokenizer" : self.model_and_tokenizer.model,
            "data_collator" : self.data_collator,
            "train_dataset" : self.tokenized_train,
            "eval_dataset" : self.tokenized_eval,
            "predict_with_generate" : True
        }

        model_type = self.model_and_tokenizer.model_type
        self.trainer = Seq2SeqTrainer(**trainer_args) if  model_type == "seq2seq" else Trainer(**trainer_args)
    
    def train(self,output_dir:str|None="./output",random_seed:int|None=None):
        """
        ### DESC
            Method for training the model.
        ### PARAMS
        * output_dir: Output model (and tokenizer) path directory (None if don't want to save).
        * random_seed: Random seed for training.
        """
        set_seed(random_seed)

        self.trainer.train()

        if output_dir != None:
            if self.trainer.is_world_process_zero():
                self.model_and_tokenizer.tokenizer.save_pretrained(output_dir)
            self.model_and_tokenizer.model.save_pretrained(save_directory=output_dir)
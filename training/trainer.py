from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from model import ABSAGenerativeModelWrapper
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
        pass
    def train(self):
        pass
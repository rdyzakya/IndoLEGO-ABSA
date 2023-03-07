import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import EvalPrediction
from transformers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from data_utils.dataset import handle_mix_sentiment, remove_duplicate_targets
from model import ABSAGenerativeModelWrapper
from evaluation import recall, precision, f1_score
from data_utils import ABSADataset, NonABSADataset, Pattern, CONSTANT_VOCAB
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
import numpy as np

class ABSAGenerativeTrainer:
    """
    Trainer for generative ABSA problem.

    How to use:

    1. Call the constructor.
    2. Call the prepare_data method.
    3. Call the compile_train_args method.
    4. Call the prepare_trainer method.
    5. Call train method.
    """
    def __init__(self,absa_model_and_tokenizer:ABSAGenerativeModelWrapper,pattern:Pattern=Pattern()):
        """"
        ### DESC
            ABSAGenerativeTrainer constructor.
        ### PARAMS
        * absa_model_and_tokenizer: ABSAGenerativeModelWrapper instance.
        """
        new_vocab = CONSTANT_VOCAB + list(pattern.mask.keys()) + list(pattern.mask.values()) + pattern.categories
        absa_model_and_tokenizer.add_vocab(new_vocab)

        self.model_and_tokenizer = absa_model_and_tokenizer
        self.pattern = pattern

    def prepare_data(self,train_dataset:Dataset,eval_dataset:Dataset=None,**encoding_args):
        """
        ### DESC
            Method for preparing data (data collator and tokenize the dataset).
        ### PARAMS
        * train_dataset: Training dataset.
        * eval_dataset: Eval dataset.
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
        else: # "causal_lm"
            causal_lm_train_input = [train_dataset["input"][i] + ' ' + train_dataset["output"][i] for i in range(len(train_dataset))]
            self.tokenized_train = tokenizer(causal_lm_train_input, **encoding_args)
            if eval_dataset != None:
                causal_lm_eval_input = [eval_dataset["input"][i] + ' ' + eval_dataset["output"][i] for i in range(len(eval_dataset))]
                self.tokenized_eval = tokenizer(causal_lm_eval_input, **encoding_args)
        
        self.train_tasks = train_dataset.data_frame.task.tolist()
        self.eval_tasks = eval_dataset.data_frame.task.tolist()
    
    def compute_metrics(self,eval_preds:EvalPrediction) -> Dict[str,float]: # MAY NOT BE SUFFICIATE FOR CAUSAL LM
        """
        ### DESC
            Method to compute the metrics.
        ### PARAMS
        * eval_preds: EvalPrediction instance from training.
        ### RETURN
        * metrics: Dictionary of metrics.
        """
        input_ids = eval_preds.inputs
        target_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions

        # In case the model returns more than the prediction logits
        if isinstance(input_ids, tuple):
            input_ids = input_ids[0]
        if isinstance(target_ids, tuple):
            target_ids = target_ids[0]
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        input_ids = np.argmax(input_ids,axis=-1) if len(input_ids.shape) == 3 else input_ids # in case not predict with generate
        target_ids = np.argmax(target_ids,axis=-1) if len(target_ids.shape) == 3 else target_ids # in case not predict with generate
        predictions = np.argmax(pred_ids,axis=-1) if len(pred_ids.shape) == 3 else pred_ids # in case not predict with generate

        inputs = self.model_and_tokenizer.tokenizer.batch_decode(inputs,skip_special_tokens=True)
        targets = self.model_and_tokenizer.tokenizer.batch_decode(targets,skip_special_tokens=True)
        predictions = self.model_and_tokenizer.tokenizer.batch_decode(predictions,skip_special_tokens=True)

        targets = [self.pattern.find_all(text,task) for text,task in zip(targets,self.eval_tasks) if task != "non_absa"]
        predictions = [self.pattern.find_all(text,task) for text,task in zip(predictions,self.eval_tasks) if task != "non_absa"]

        per_task_targets, per_task_predictions = self.seperate_target_prediction_per_task(predictions, targets)
        
        metrics = {}

        metrics["overall_recall"] = recall(predictions,targets)
        metrics["overall_precision"] = precision(predictions,targets)
        metrics["overall_f1_score"] = f1_score(predictions,targets)

        for task in per_task_targets.keys():
            metrics[f"{task}_recall"] = recall(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_precision"] = precision(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_f1_score"] = f1_score(per_task_predictions[task],per_task_targets[task])
        
        return metrics

    def seperate_target_prediction_per_task(self,predictions:List[List[Dict]],targets:List[List[Dict]]) -> tuple[Dict[str,List],Dict[str,List]]:
        per_task_targets = {}
        per_task_predictions = {}
        for target, prediction, task in zip(targets,predictions,self.eval_tasks):
            if task not in per_task_targets.keys():
                per_task_targets[task] = []
            if task not in per_task_predictions.keys():
                per_task_predictions[task] = []
            per_task_targets[task].append(target)
            per_task_predictions[task].append(prediction)
        return per_task_targets, per_task_predictions
        
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
            "predict_with_generate" : True,
            "include_inputs_for_metrics" : True,
            "compute_metrics" : self.compute_metrics
        }

        model_type = self.model_and_tokenizer.model_type
        self.trainer = Seq2SeqTrainer(**trainer_args) if  model_type == "seq2seq" else Trainer(**trainer_args)
    
    def train(self,output_dir:str="./output",random_seed:int=None):
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
    
    def predict_absa(self,dataset:ABSADataset,task_tree:Dict={"acos" : {"ao" : [],"as" : [],"aos" : ['a']}},device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512):
        predictions = {}
        if isinstance(task_tree,Dict):
            for main_task, children_task in task_tree.items():
                predictions[main_task] = self.predict_absa_per_task(dataset,main_task,children_task,device,batch_size,encoding_args,decoding_args,max_len)
        else:
            for main_task in task_tree:
                predictions[main_task] = self.predict_absa_per_task(dataset,main_task,[],device,batch_size,encoding_args,decoding_args,max_len)
        return predictions

    def predict_absa_per_task(self,dataset:ABSADataset,task:str="aos",children_task:Dict={"ao" : ['a'], 'a' : []},device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512):        
        # Extraction for main task
        # Recursive
        predictions = self.predict_absa_per_task_per_paradigm(dataset,[],task,"extraction",device,batch_size,encoding_args,decoding_args,max_len)
        if isinstance(children_task,Dict):
            for child_task in children_task.keys():
                child_predictions = self.predict_absa_per_task(dataset,child_task,children_task[child_task],device,batch_size,encoding_args,decoding_args,max_len)
                self.add_imputation_predictions(dataset, task, device, batch_size, encoding_args, decoding_args, max_len, predictions, child_predictions)
        else: # List
            for child_task in children_task:
                child_predictions = self.predict_absa_per_task_per_paradigm(dataset,[],child_task,"extraction",device,batch_size,encoding_args,decoding_args,max_len)
                self.add_imputation_predictions(dataset, task, device, batch_size, encoding_args, decoding_args, max_len, predictions, child_predictions)
        return predictions

    def add_imputation_predictions(self, dataset, task, device, batch_size, encoding_args, decoding_args, max_len, predictions, child_predictions):
        imputation_predictions = self.predict_absa_per_task_per_paradigm(dataset,child_predictions,task,"imputation",device,batch_size,encoding_args,decoding_args,max_len)
        assert len(predictions) == len(imputation_predictions)
        for i_row in range(len(predictions)):
            pred = predictions[i_row] + imputation_predictions[i_row]
            pred = handle_mix_sentiment(pred)
            pred = remove_duplicate_targets(pred)
            predictions[i_row] = pred

    def predict_absa_per_task_per_paradigm(self,dataset:ABSADataset,incomplete_targets:List[List[Dict]],task:str='a',paradigm:str="extraction",device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512):
        predictions = []
        # Build the test dataset
        test_dataset = dataset.build_test_data(task,paradigm,incomplete_targets)
        # Tokenize the input
        tokenizer = self.model_and_tokenizer.tokenizer
        if self.model_type == "seq2seq":
            tokenized_test = tokenizer(test_dataset["input"], text_target=test_dataset["output"], **encoding_args)
        else: # "causal_lm"
            causal_lm_test_input = [test_dataset["input"][i] + ' ' + test_dataset["output"][i] for i in range(len(test_dataset))]
            tokenized_test = tokenizer(causal_lm_test_input, **encoding_args)
        # Move the model to device
        self.model_and_tokenizer.to(device)
        # Data loader
        data_loader = torch.utils.data.DataLoader(tokenized_test["input_ids"],
                            batch_size=batch_size,shuffle=False)
        # Predict
        model = self.model_and_tokenizer.model
        tensor_predictions = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                tensor_predictions.extend(model.generate(input_ids=batch.to(device),max_length=max_len))
        decoded_predictions = tokenizer.batch_decode(tensor_predictions,**decoding_args)
        for i_pred in range(len(decoded_predictions)):
            new_prediction = self.pattern.find_all(decoded_predictions[i_pred],task)
            new_prediction = handle_mix_sentiment(new_prediction)
            new_prediction = remove_duplicate_targets(new_prediction)
            predictions.append(new_prediction)
        
        return predictions

    def predict_non_absa(self):
        pass
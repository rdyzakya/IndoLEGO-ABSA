import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import EvalPrediction
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from data_utils.dataset import handle_mix_sentiment, remove_duplicate_targets
from model import ABSAGenerativeModelWrapper
from evaluation import recall, precision, f1_score, summary_score
from data_utils import ABSADataset, NonABSADataset, Pattern, CONSTANT_VOCAB
from datasets import Dataset
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

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
    def __init__(self,absa_model_and_tokenizer:ABSAGenerativeModelWrapper,pattern:Pattern,do_train:bool=True,do_eval:bool=False):
        """"
        ### DESC
            ABSAGenerativeTrainer constructor.
        ### PARAMS
        * absa_model_and_tokenizer: ABSAGenerativeModelWrapper instance.
        * pattern: Pattern instance.
        * do_train: Do training.
        * do_eval: Do validation.
        """
        new_vocab = CONSTANT_VOCAB + list(pattern.mask.keys()) + list(pattern.mask.values()) + pattern.categories
        absa_model_and_tokenizer.add_vocab(new_vocab)

        self.model_and_tokenizer = absa_model_and_tokenizer
        self.pattern = pattern
        self.do_train = do_train
        self.do_eval = do_eval

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

        def create_clm(row,train=False):
            # if train:
            #     return {
            #             "input" : row["input"] + ' ' + tokenizer.sep_token + ' ' + row["output"] + ' ' + tokenizer.eos_token,
            #             "output" : row["input"] + ' ' + tokenizer.sep_token + ' ' + row["output"] + ' ' + tokenizer.eos_token
            #         }
            # return {
            #         "input" : row["input"] + ' ' + tokenizer.sep_token,
            #         "output" : row["input"] + ' ' + tokenizer.sep_token + ' ' + row["output"] + ' ' + tokenizer.eos_token
            #     }
            return {
                        "causal_lm_input" : row["input"] + ' ' + tokenizer.sep_token + ' ' + row["output"] + ' ' + tokenizer.eos_token,
                    }
          
        if model_type == "causal_lm":
          if self.do_train:
            train_dataset = train_dataset.map(lambda x:create_clm(x,True))
          if self.do_eval:
            eval_dataset = eval_dataset.map(lambda x:create_clm(x,False))

        def encode(dataset):
            if model_type == "seq2seq":
                result = tokenizer(dataset["input"], text_target=dataset["output"], **encoding_args)
                return result
            return tokenizer(dataset["causal_lm_input"],**encoding_args)

        # Prepare data collator
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if self.model_and_tokenizer.model_type == "seq2seq" else DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        
        # Encode the input and output
        if self.do_train:
            self.tokenized_train = train_dataset.map(encode,batched=True,remove_columns=train_dataset.column_names)
            self.train_tasks = train_dataset["task"]
        
        if self.do_eval:
            self.tokenized_eval = eval_dataset.map(encode,batched=True,remove_columns=eval_dataset.column_names)
            self.eval_tasks = eval_dataset["task"]
    
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
        prediction_ids = np.argmax(pred_ids,axis=-1) if len(pred_ids.shape) == 3 else pred_ids # in case not predict with generate

        input_ids = [[token for token in row if token != -100] for row in input_ids]
        target_ids = [[token for token in row if token != -100] for row in target_ids]
        prediction_ids = [[token for token in row if token != -100] for row in prediction_ids]

        inputs = self.model_and_tokenizer.tokenizer.batch_decode(input_ids,skip_special_tokens=True)
        targets = self.model_and_tokenizer.tokenizer.batch_decode(target_ids,skip_special_tokens=True)
        predictions = self.model_and_tokenizer.tokenizer.batch_decode(prediction_ids,skip_special_tokens=True)

        print("[DEBUG LOOKUP INPUT OUTPUT]")
        print(">> INPUT:",inputs[10:12])
        print(">> TARGETS:",targets[10:12])
        print(">> OUTPUT:",predictions[10:12])
        print("[END]")

        targets = [self.pattern.find_all(text,task) for text,task in zip(targets,self.eval_tasks) if task != "non_absa"]
        predictions = [self.pattern.find_all(text,task) for text,task in zip(predictions,self.eval_tasks) if task != "non_absa"]


        per_task_targets, per_task_predictions = self.seperate_target_prediction_per_task(predictions, targets)
        
        metrics = {}

        metrics["overall_recall"] = recall(predictions,targets)
        metrics["overall_precision"] = precision(predictions,targets)
        metrics["overall_f1_score"] = f1_score(predictions,targets)

        for task in per_task_targets.keys():
            if task == "non_absa":
                continue
            metrics[f"{task}_recall"] = recall(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_precision"] = precision(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_f1_score"] = f1_score(per_task_predictions[task],per_task_targets[task])
        
        return metrics

    def seperate_target_prediction_per_task(self,predictions:List[List[Dict]],targets:List[List[Dict]]) -> Tuple[Dict[str,List],Dict[str,List]]:
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
    
    def preprocess_logits_for_metrics(self, logits, labels):
        pred_logits = logits[0] if isinstance(logits,tuple) else logits
        pred_ids = torch.argmax(pred_logits, dim=-1)
        return pred_ids, labels
        
    def compile_train_args(self,train_args_dict:Dict):
        """
        ### DESC
            Method to load training arguments.
        ### PARAMS
        * train_args_dict: Training arguments (dictionary).
        """
        train_args_dict.update({
            "include_inputs_for_metrics" : True
        })
        # if self.model_and_tokenizer.model_type == "seq2seq":
        #     train_args_dict.update(
        #         {"predict_with_generate" : True,}
        #     )
        self.training_args = Seq2SeqTrainingArguments(**train_args_dict) # if self.model_and_tokenizer.model_type == "seq2seq" else TrainingArguments(**train_args_dict)

    def prepare_trainer(self):
        """
        ### DESC
            Method for preparing trainer.
        """
        trainer_args = {
            "model" : self.model_and_tokenizer.model,
            "args" : self.training_args,
            "tokenizer" : self.model_and_tokenizer.model,
            "data_collator" : self.data_collator
        }

        if self.train:
            trainer_args.update({
                "train_dataset" : self.tokenized_train
            })
        if self.do_eval:
            trainer_args.update({
                "eval_dataset" : self.tokenized_eval,
                "compute_metrics" : self.compute_metrics
            })

        model_type = self.model_and_tokenizer.model_type

        # if self.model_and_tokenizer.model_type == "causal_lm":
        trainer_args["preprocess_logits_for_metrics"] = self.preprocess_logits_for_metrics

        self.trainer = Seq2SeqTrainer(**trainer_args) # if  model_type == "seq2seq" else Trainer(**trainer_args)
    
    def train(self,output_dir:str="./output",random_seed:int=None):
        """
        ### DESC
            Method for training the model.
        ### PARAMS
        * output_dir: Output model (and tokenizer) path directory (None if don't want to save).
        * random_seed: Random seed for training.
        """
        if isinstance(random_seed,int):
            set_seed(random_seed)

        train_result = self.trainer.train()
        train_metrics = train_result.metrics
        self.trainer.log_metrics("train", train_metrics)
        self.trainer.save_metrics("train", train_metrics)

        if self.do_eval:
            eval_metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)

        if output_dir != None:
            if self.trainer.is_world_process_zero():
                self.model_and_tokenizer.tokenizer.save_pretrained(output_dir)
            self.model_and_tokenizer.model.save_pretrained(save_directory=output_dir)
    
    def predict_absa(self,dataset:ABSADataset,task_tree:Dict={"acos" : {"ao" : [],"as" : [],"aos" : ['a']}},device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512) -> Tuple[Dict,Dict,Dict]:
        """
        ### DESC
            Method for predicting absa tasks.
        ### PARAMS
        * dataset: ABSADataset instance.
        * task_tree: List of task names or in a dictionary form with a tree style.
        * device: Torch device instance.
        * batch_size: Batch size.
        * encoding_args: Dictionary containing encoding key word arguments.
        * decoding_args: Dictionary containing decoding key word arguments.
        * max_len: Maximum length of the decoded result.
        ### RETURN
        * ABSA targets for all the task contained in the task tree.
        * Decoded ABSA predictions.
        * Summary scores.
        """
        # Move the model to device
        self.model_and_tokenizer.to(device)
        predictions = {}
        # decoded_predictions = {}
        all_str_preds = pd.DataFrame()
        summary = {}
        if isinstance(task_tree,Dict):
            for main_task, children_task in task_tree.items():
                predictions[main_task], _ = self.predict_absa_per_task(dataset=dataset,all_str_preds=all_str_preds,task=main_task,
                                                                children_task=children_task,device=device,
                                                                batch_size=batch_size,encoding_args=encoding_args,
                                                                decoding_args=decoding_args,max_len=max_len)
                blank_incomplete_targets = [[] for n in range(len(dataset))]
                targets = dataset.build_test_data(main_task,"extraction",blank_incomplete_targets)["target"]
                summary[main_task] = summary_score(predictions[main_task],targets)
        else:
            for main_task in task_tree:
                predictions[main_task], _ = self.predict_absa_per_task(dataset=dataset,all_str_preds=all_str_preds,task=main_task,
                                                                children_task=[],device=device,
                                                                batch_size=batch_size,encoding_args=encoding_args,
                                                                decoding_args=decoding_args,max_len=max_len)
                blank_incomplete_targets = [[] for n in range(len(dataset))]
                targets = dataset.build_test_data(main_task,"extraction",blank_incomplete_targets)["target"]
                summary[main_task] = summary_score(predictions[main_task],targets)
        self.model_and_tokenizer.to(torch.device("cpu"))
        return predictions, summary, all_str_preds

    def predict_absa_per_task(self,dataset:ABSADataset,all_str_preds:pd.DataFrame,task:str="aos",children_task:Dict={"ao" : ['a'], 'a' : []},device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512) -> Tuple[List[List[Dict]],List[List[str]]]:
        """
        ### DESC
            Method to predict absa task in a post order traversal manner. The lower level will help the imputation process for the upper level.
        ### PARAMS
        * dataset: ABSADataset instance.
        * task: Main task name.
        * children_task: Task children in a form of list or dictionary (tree style).
        * device: Torch device instance.
        * batch_size: Batch size.
        * encoding_args: Dictionary containing encoding key word arguments.
        * decoding_args: Dictionary containing decoding key word arguments.
        * max_len: Maximum length of the decoded result.
        ### RETURN
        * ABSA targets for the designated task.
        * ABSA decoded predictions.
        """
        # Extraction for main task
        # Recursive

        # Do extraction for the main task
        blank_incomplete_targets = [[] for n in range(len(dataset))]
        predictions, str_preds = self.predict_absa_per_task_per_paradigm(dataset,blank_incomplete_targets,task,"extraction",device,batch_size,encoding_args,decoding_args,max_len)
        all_str_preds[f"{task}_extraction"] = str_preds
        if isinstance(children_task,Dict):
            for child_task in children_task.keys():
                # Do extraction for the children task
                child_predictions, child_str_preds = self.predict_absa_per_task(dataset,all_str_preds,child_task,children_task[child_task],device,batch_size,encoding_args,decoding_args,max_len)
                all_str_preds[f"{child_task}_extraction"] = child_str_preds
                # From the children task result, impute the tuples resulting the main task
                imputation_predictions, imputation_str_preds = self.predict_absa_per_task_per_paradigm(dataset,child_predictions,task,"imputation",device,batch_size,encoding_args,decoding_args,max_len)
                all_str_preds[f"{task}_imputation_{child_task}"] = imputation_str_preds
                self.add_imputation_predictions(predictions, imputation_predictions)
        else: # List
            for child_task in children_task:
                # Do extraction for the children task
                blank_incomplete_targets = [[] for n in range(len(dataset))]
                child_predictions, child_str_preds = self.predict_absa_per_task_per_paradigm(dataset,blank_incomplete_targets,child_task,"extraction",device,batch_size,encoding_args,decoding_args,max_len)
                all_str_preds[f"{child_task}_extraction"] = child_str_preds
                # From the children task result, impute the tuples resulting the main task
                imputation_predictions, imputation_str_preds = self.predict_absa_per_task_per_paradigm(dataset,child_predictions,task,"imputation",device,batch_size,encoding_args,decoding_args,max_len)
                all_str_preds[f"{task}_imputation_{child_task}"] = imputation_str_preds
                self.add_imputation_predictions(predictions, imputation_predictions)
                # self.add_imputation_predictions(dataset, predictions, child_predictions, task, device, batch_size, encoding_args, decoding_args, max_len)
        return predictions, str_preds

    def add_imputation_predictions(self, predictions:List[List[Dict]], imputation_predictions:List[List[Dict]]):
        """
        ### DESC
            Method to add imputation predictions to the prediction list.
        ### PARAMS
        * dataset: ABSADataset instance.
        * predictions: Predictions result.
        * child_predictions: Predictions for the imputation.
        * decoded_predictions: Decoded predictions.
        * task: Main task name.
        * device: Torch device instance.
        * batch_size: Batch size.
        * encoding_args: Dictionary containing encoding key word arguments.
        * decoding_args: Dictionary containing decoding key word arguments.
        * max_len: Maximum length of the decoded result.
        """
        assert len(predictions) == len(imputation_predictions)
        for i_row in range(len(predictions)):
            pred = predictions[i_row] + imputation_predictions[i_row]
            pred = handle_mix_sentiment(pred)
            pred = remove_duplicate_targets(pred)
            predictions[i_row] = pred
            # decoded_predictions[i_row] = decoded_predictions[i_row] + decoded_imputation[i_row]

    def predict_absa_per_task_per_paradigm(self,dataset:ABSADataset,incomplete_targets:List[List[Dict]],task:str='a',paradigm:str="extraction",device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512) -> Tuple[List[List[Dict]],List[List[str]]]:
        """
        ### DESC
            Method to predict an absa task with a certain paradigm (extraction or imputation).
        ### PARAMS
        * dataset: ABSADataset instance.
        * incomplete_targets: Incomplete targets for imputation.
        * task: Main task name.
        * device: Torch device instance.
        * batch_size: Batch size.
        * encoding_args: Dictionary containing encoding key word arguments.
        * decoding_args: Dictionary containing decoding key word arguments.
        * max_len: Maximum length of the decoded result.
        ### RETURN
        * ABSA targets for the designated task with the designated paradigm.
        * Decoded predictions.
        """
        predictions = []
        # Build the test dataset
        test_dataset = dataset.build_test_data(task,paradigm,incomplete_targets)
        # Tokenize the input
        tokenizer = self.model_and_tokenizer.tokenizer
        if self.model_and_tokenizer.model_type == "seq2seq":
            tokenized_test = tokenizer(test_dataset["input"], text_target=test_dataset["output"], **encoding_args)
        else: # "causal_lm"
            # causal_lm_test_input = [test_dataset["input"][i] + ' ' + test_dataset["output"][i] for i in range(len(test_dataset))]
            def create_clm(row):
                return {"causal_lm_input" : row["input"] + ' ' + tokenizer.sep_token}
            test_dataset = test_dataset.map(create_clm)
            tokenized_test = tokenizer(test_dataset["causal_lm_input"], **encoding_args)
        # Predict
        str_preds = self.generate_predictions(tokenized_test,device,batch_size,max_len,decoding_args)
        for i_pred in range(len(str_preds)):
            new_prediction = self.pattern.find_all(str_preds[i_pred],task)
            new_prediction = handle_mix_sentiment(new_prediction)
            new_prediction = remove_duplicate_targets(new_prediction)
            predictions.append(new_prediction)
            # decoded_predictions[i_pred] = [decoded_predictions[i_pred]] # Becomes List[List[str]]
        
        return predictions, str_preds

    def generate_predictions(self,tokenized:torch.Tensor,device:torch.device=torch.device("cpu"),batch_size:int=16,max_len:int=512,decoding_args:Dict={}) -> List[str]:
        """
        ### DESC
            Method to generate predictions using generative model.
        ### PARAMS
        * tokenized: Tokenized text.
        * device: Torch device instance.
        * batch_size: Batch size.
        * max_len: Maximum length of the decoded result.
        * decoding_args: Dictionary containing decoding key word arguments.
        ### RETURN
        * Decoded predictions.
        """
        # Data loader
        input_ids_data_loader = torch.utils.data.DataLoader(tokenized["input_ids"],
                            batch_size=batch_size,shuffle=False)
        attention_mask_data_loader = torch.utils.data.DataLoader(tokenized["attention_mask"],
                            batch_size=batch_size,shuffle=False)
        # Predict
        model = self.model_and_tokenizer.model
        tokenizer = self.model_and_tokenizer.tokenizer
        tensor_predictions = []
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(zip(input_ids_data_loader,attention_mask_data_loader)):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                tensor_predictions.extend(model.generate(input_ids=input_ids,attention_mask=attention_mask,max_length=max_len,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id).cpu())
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
        tensor_predictions = [[token for token in row if token != -100] for row in tensor_predictions]
        predictions = tokenizer.batch_decode(tensor_predictions,**decoding_args)
        return predictions
    
    def predict_non_absa(self,dataset:List[NonABSADataset],device:torch.device=torch.device("cpu"),batch_size:int=16,encoding_args:Dict={},decoding_args:Dict={},max_len:int=512) -> List[str]:
        """
        ### DESC
            Mthod to predict non absa datasets.
        ### PARAMS
        * dataset: List of NonABSADataset instance.
        * device: Torch device instance.
        * batch_size: Batch size.
        * encoding_args: Dictionary containing encoding key word arguments.
        * decoding_args: Dictionary containing decoding key word arguments.
        * max_len: Maximum length of the decoded result.
        ### RETURN
        * Decoded predictions for non absa task.
        """
        try:
            test_dataset = [ds.build_data().to_pandas() for ds in dataset]
            test_dataset = pd.concat(test_dataset,axis=0)
            tokenizer = self.model_and_tokenizer.tokenizer
            tokenized_test = tokenizer(test_dataset["input"].values.tolist(), **encoding_args)
            return self.generate_predictions(tokenized_test,device,batch_size,max_len,decoding_args)
        except:
            return []
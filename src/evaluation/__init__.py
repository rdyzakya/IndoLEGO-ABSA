# 🤗
from .metrics import *

from transformers import EvalPrediction, AutoTokenizer
from evaluation import recall, precision, f1_score, summary_score
from typing import List, Dict, Tuple
import numpy as np

def seperate_target_prediction_per_task(predictions:List[List[Dict]],targets:List[List[Dict]],tasks:List) -> Tuple[Dict[str,List],Dict[str,List]]:
    per_task_targets = {}
    per_task_predictions = {}
    for target, prediction, task in zip(targets,predictions,tasks):
        if task not in per_task_targets.keys():
            per_task_targets[task] = []
        if task not in per_task_predictions.keys():
            per_task_predictions[task] = []
        per_task_targets[task].append(target)
        per_task_predictions[task].append(prediction)
    return per_task_targets, per_task_predictions

def preprocess_eval_preds(eval_preds:EvalPrediction,decoding_args:Dict[str,str],tokenizer:AutoTokenizer):
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

    inputs = tokenizer.batch_decode(input_ids,**decoding_args)
    targets = tokenizer.batch_decode(target_ids,**decoding_args)
    predictions = tokenizer.batch_decode(prediction_ids,**decoding_args)

    return inputs, targets, predictions

def compute_metrics(catch_answer,eval_preds:EvalPrediction,decoding_args:Dict[str,str],tokenizer:AutoTokenizer,tasks:List) -> Dict[str,float]: # MAY NOT BE SUFFICIATE FOR CAUSAL LM
        """
        ### DESC
            Method to compute the metrics.
        ### PARAMS
        * catch_answer: catch answer function.
        * eval_preds: EvalPrediction instance from training.
        * decoding_args: Decoding arguments.
        ### RETURN
        * metrics: Dictionary of metrics.
        """
        inputs, targets, predictions = preprocess_eval_preds(eval_preds,decoding_args,tokenizer)

        print("INPUTS >>",inputs[0])
        print("TARGETS >>",targets[0])
        print("PREDS >>",predictions[0])

        targets = [catch_answer(out,task,inputs) for out,task,inputs in zip(targets,tasks,inputs) if task != "non_absa"]
        predictions = [catch_answer(out,task,inputs) for out,task,inputs in zip(predictions,tasks,inputs) if task != "non_absa"]

        per_task_targets, per_task_predictions = seperate_target_prediction_per_task(predictions, targets, tasks)
        
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
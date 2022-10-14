from typing import List, Tuple, Dict
import nltk

import pandas as pd
import numpy as np
from transformers import TrainerCallback

from preprocessing import postprocess_gaste_output

import model_types

def one_target_edit_score(dict_true : Dict, dict_pred : Dict) -> float:
    """
    [DESC]
        Compute the edit score between two tuples
    [PARAMS]
        dict_true : dict
            True target
        dict_pred : dict
            Prediction target
    [RETURNS]
        score : float
    """
    score = 0
    assert len(dict_pred) > 0 and len(dict_true) > 0 # If error, then the dictionary is empty
    assert sorted(dict_true.keys()) == sorted(dict_pred.keys()) # If error, then different task
    keys = dict_true.keys()
    # check polarity
    if "sentiment" in keys:
        if dict_true["sentiment"] != dict_pred["sentiment"]:
            return 0
    # check the aspect
    if "aspect" in keys:
        levenshtein_distance = nltk.edit_distance(dict_true["aspect"], dict_pred["aspect"])
        denom = max(len(dict_true["aspect"]), len(dict_pred["aspect"]))
        score += 1 - (levenshtein_distance / denom)
    # check the opinion marker
    if "opinion" in keys:
        levenshtein_distance = nltk.edit_distance(dict_true["opinion"], dict_pred["opinion"])
        denom = max(len(dict_true["opinion"]), len(dict_pred["opinion"]))
        score += 1 - (levenshtein_distance / denom)
    if "aspect" in keys and "opinion" in keys:
        score = score / 2
    return score

def one_row_edit_score(y_true : List[Dict], y_pred : List[Dict]) -> float:
    """
    [DESC]
        Function to calculate the edit score per row/instance data
    [PARAMS]
        y_true : List[Dict]
            True targets
        y_pred : List[Dict]
            Prediction targets
    [RETURNS]
        score : dict
    """
    len_y_true = len(y_true)
    len_y_pred = len(y_pred)

    if len_y_true == 0 or len_y_pred == 0:
        return 0

    score_matrix = []
    for i in range(len_y_true):
        score_matrix.append([])
        for j in range(len_y_pred):
            score_matrix[i].append(one_target_edit_score(y_true[i],y_pred[j]))
    # score = np.max(score_matrix, axis=1).sum() if score_type == "recall" else np.max(score_matrix, axis=0).sum()
    return {"recall" : np.max(score_matrix, axis=1).sum(), "precision" : np.max(score_matrix, axis=0).sum()}

def evaluate(pred_pt : List[List[Dict]], gold_pt : List[List[Dict]]) -> Dict[str,float]:
    """
    [DESC]
        Function to compute F1 scores with pred and gold pairs/triplets
        The input needs to be already processed
    [PARAMS]
        pred_pt : List[List[Dict]]
        gold_pt : List[List[Dict]]
    [RETURNS]
        scores : dict
    """
    assert len(pred_pt) == len(gold_pt) # If this is error then the length of sents and labels are different
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0
    total_edit_score_precision, total_edit_score_recall = 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        
        edit_score = one_row_edit_score(gold_pt[i], pred_pt[i])
        total_edit_score_recall += edit_score["recall"]
        total_edit_score_precision += edit_score["precision"]

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    
    edit_score_recall = (total_edit_score_recall / n_gold) if n_gold != 0 else 0
    edit_score_precision = (total_edit_score_precision / n_pred) if n_pred != 0 else 0
    edit_score_f1 = 2 * edit_score_precision * edit_score_recall / (edit_score_precision + edit_score_recall) if edit_score_precision != 0 or edit_score_recall != 0 else 0
    
    scores = {'precision': precision, 'recall': recall, 'f1': f1, 'edit_recall' : edit_score_recall, 'edit_precision' : edit_score_precision, 'edit_f1': edit_score_f1}
    return scores

def absa_compute_metrics(eval_preds,text_dataset,tokenizer,model_type,prompt_option=0,quote=True,quote_with_space=True,**kwargs):
    if model_type not in model_types.seq2seq and model_type not in model_types.lm:
        raise ValueError(f"Model types available : {model_types.seq2seq + model_types.lm}")
    print("Computing evaluation metrics..")
    preds, labels = eval_preds

    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.argmax(preds,axis=-1) if len(preds.shape) == 3 else preds # in case not predict with generate
    
    texts = text_dataset["text"]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, **kwargs)
    inverse_stringified_preds = postprocess_gaste_output(texts,decoded_preds,
    model_type,tokenizer,prompt_option,quote,quote_with_space,**kwargs)
    print("Prediction sample:",inverse_stringified_preds[0])
    real_labels = text_dataset["target"]
    for i in range(len(real_labels)):
        for j in range(len(real_labels[i])):
            real_labels[i][j] = tuple(real_labels[i][j])
    print("Labels sample:",real_labels[0])

    metrics = evaluate(texts,inverse_stringified_preds,real_labels)
    
    return metrics

class EvaluationCallback(TrainerCallback):
    def __init__(self,output_dir,**kwargs):
        self.output_dir = output_dir
        self.all_metrics = []

    def on_evaluate(self,args,state,controls,metrics,**kwargs):
        self.all_metrics.append(metrics)
    
    def on_train_end(self,args,state,controls,**kwargs):
        list_metrics = self.all_metrics
        for i in range(len(list_metrics)):
            list_metrics[i]["epoch"] = i+1
        df_metrics = pd.DataFrame(list_metrics)
        df_metrics.to_csv(self.output_dir,index=False)
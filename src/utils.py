import random
import numpy as np
import torch
import evaluation

def set_seed(seed: int):
    """Set the seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_token_clm(model, tokenizer):
    resize = False
    if tokenizer.pad_token == None:
        pad_token = "<|pad|>"
        tokenizer.add_tokens([pad_token])
        tokenizer.add_special_tokens({"pad_token": pad_token})
        resize = True
    if tokenizer.eos_token == None:
        eos_token = "<|endoftext|>"
        tokenizer.add_tokens([eos_token])
        tokenizer.add_special_tokens({"eos_token": eos_token})
        resize = True
    if tokenizer.sep_token == None:
        sep_token = "<|sep|>"
        tokenizer.add_tokens([sep_token])
        tokenizer.add_special_tokens({"sep_token": sep_token})
        resize = True
    if resize:
       model.resize_token_embeddings(len(tokenizer))

def preprocess_logits_for_metrics(logits, targets):
    pred_logits = logits[0] if isinstance(logits,tuple) else logits
    pred_ids = torch.argmax(pred_logits, dim=-1)
    return pred_ids, targets

def get_task(se_order):
    task = sorted(se_order)
    task = ''.join(se_order)
    return task

def seperate_target_prediction_per_task(predictions, targets, se_order):
    per_task_targets = {}
    per_task_predictions = {}
    for target, prediction, so in zip(targets,predictions,se_order):
        task = get_task(so)
        if task not in per_task_targets.keys():
            per_task_targets[task] = []
        if task not in per_task_predictions.keys():
            per_task_predictions[task] = []
        per_task_targets[task].append(target)
        per_task_predictions[task].append(prediction)
    return per_task_targets, per_task_predictions

def preprocess_eval_preds(eval_preds, decoding_args ,tokenizer):
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

def compute_metrics(catch_answer, eval_preds, decoding_args, tokenizer, se_order):
        inputs, targets, predictions = preprocess_eval_preds(eval_preds,decoding_args,tokenizer)

        print("INPUTS >>",inputs[0])
        print("TARGETS >>",targets[0])
        print("PREDS >>",predictions[0])

        targets = [catch_answer(out,task,inputs) for out,task,inputs in zip(targets,se_order,inputs) if task != "non_absa"]
        predictions = [catch_answer(out,task,inputs) for out,task,inputs in zip(predictions,se_order,inputs) if task != "non_absa"]

        per_task_targets, per_task_predictions = seperate_target_prediction_per_task(predictions, targets, se_order)
        
        metrics = {}

        metrics["overall_recall"] = evaluation.recall(predictions,targets)
        metrics["overall_precision"] = evaluation.precision(predictions,targets)
        metrics["overall_f1_score"] = evaluation.f1_score(predictions,targets)

        for task in per_task_targets.keys():
            if task == "non_absa":
                continue
            metrics[f"{task}_recall"] = evaluation.recall(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_precision"] = evaluation.precision(per_task_predictions[task],per_task_targets[task])
            metrics[f"{task}_f1_score"] = evaluation.f1_score(per_task_predictions[task],per_task_targets[task])
        
        return metrics
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: absa
#     language: python
#     name: python3
# ---

# %%
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
n_gpu = torch.cuda.device_count()

# %%
import pandas as pd

# %%
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
sys.path.append("../../../src/")
import data_utils

# %% [markdown]
# # Dataset Utilities

# %%
william_dir = dict(
    hotel = "../../../data/absa/id/william"
)

william = dict(
    hotel = dict(
        train = data_utils.read_data(path=william_dir["hotel"] + "/train.txt",
                                     target_format="aos"),
        val = data_utils.read_data(path=william_dir["hotel"] + "/dev.txt",
                                     target_format="aos"),
        test = data_utils.read_data(path=william_dir["hotel"] + "/test.txt",
                                     target_format="aos")
    )
)

# %% [markdown]
# # Data Preprocessing 1

# %% [markdown]
# 1. AOS (ASTE)
#     * AO
#     * AS
#     * A
#     * O
#
# 2. ACS (TASD)
#     * AS
#     * CS
#     * A
#     * C
#
# 3. ACOS
#     * AO
#     * AS
#     * CS
#     * A
#     * O
#     * C

# %%
task_tree = {
    "oas" : ["oas","oa","as",'a','o'],
    "asc" : ["asc","as","sc",'a','c'],
    "oasc" : ["oasc","oa","as","sc",'a','o','c']
}

all_task = []
for k,v1 in task_tree.items():
    if k not in all_task:
        all_task.append(k)
    for v2 in v1:
        if v2 not in all_task:
            all_task.append(v2)

print(all_task)


# %%
def reduce_num_targets(num_targets,og_format,reduced_format):
    result = []
    og_format = list(og_format)
    reduced_format = list(reduced_format)
    rf_index = [og_format.index(el) for el in reduced_format]
    result = [tuple(nt[i] for i in rf_index) for nt in num_targets]
    return result

def remove_duplicates_num_targets(num_targets):
    result = []
    for el in num_targets:
        if el not in result:
            result.append(el)
    return result


# %%
from copy import deepcopy

# William (AOS ID)
william_intermediate = dict()

for domain, v1 in william.items():
    william_intermediate[domain] = dict()
    for task in ["oas"] + task_tree["oas"]:
        william_intermediate[domain][task] = dict()
        for split in v1.keys():
            ds = william[domain][split]
            ds_copy = deepcopy(ds)
            for i in range(len(ds_copy)):
                ## TARGET
                # Reduce
                ds_copy[i]["target"] = data_utils.reduce_targets(ds_copy[i]["target"],task)
                # Remove Duplicates
                ds_copy[i]["target"] = data_utils.remove_duplicate_targets(ds_copy[i]["target"])
                ## NUM TARGETS
                ds_copy[i]["num_targets"] = reduce_num_targets(ds_copy[i]["num_targets"],"aos",task)
                ds_copy[i]["num_targets"] = remove_duplicates_num_targets(ds_copy[i]["num_targets"])
            william_intermediate[domain][task][split] = ds_copy


# %% [markdown]
# # Answer Engineering

# %%
# def construct_answer(targets,se_order):
#     if len(targets) == 0:
#         return "NULL"
#     result = []
#     for t in targets:
#         constructed_t = []
#         for se in se_order:
#             element = t[data_utils.SENTIMENT_ELEMENT[se]]
#             for k, v in added_tokens.items():
#                 element = element.replace(k,v)
#             constructed_t.append(element)
#         constructed_t = " , ".join(constructed_t)
#         constructed_t = f"( {constructed_t} )"
#         result.append(constructed_t)
#     result = " ; ".join(result)
#     return result

def construct_answer(num_targets):
    if len(num_targets) == 0:
        return "NULL"
    result = []
    for nt in num_targets:
        for el in nt:
            if isinstance(el,list):
                result.append(str(el[0])) # start index
                result.append(str(el[-1])) # end index
            else:
                result.append(el)
    return ','.join(result)


# %%
william_intermediate["hotel"]["oas"]["train"][2]

# %%
construct_answer(william_intermediate["hotel"]["oas"]["train"][0]["num_targets"])


# %% [markdown]
# # Prompt Engineering

# %%
def construct_prompt(text,se_order):
    prompt = []
    for se in se_order:
        if se == 'o' or se == 'a':
            name = data_utils.SENTIMENT_ELEMENT[se]
            start_index = name + "_start"
            end_index = name + "_end"
            prompt.append(start_index)
            prompt.append(end_index)
        else:
            prompt.append(data_utils.SENTIMENT_ELEMENT[se])
    prompt = ",".join(prompt)
    # prompt = f"( {prompt} )"
    # masked_text = text
    # for k, v in added_tokens.items():
    #     masked_text = masked_text.replace(k,v)
    result = text + " | " + prompt
    return result


# %%
construct_prompt(william_intermediate["hotel"]["oas"]["train"][0]["text"],"oas")

# %% [markdown]
# # Answer Catch

# %%
import re

def catch_answer(output,se_order,text):
    splitted_text = text.split()
    if output == "NULL":
        return []
    result = []
    splitted_output = output.split(',')
    splitted_output = [el.strip() for el in splitted_output]

    chunk_size = 0
    for se in se_order:
        if se == 'o' or se == 'a':
            chunk_size += 2
        else:
            chunk_size += 1

    chunks = [
        splitted_output[i:i+chunk_size] for i in range(0,len(splitted_output),chunk_size)
    ]

    chunks = [el for el in chunks if len(el) == chunk_size]

    for el in chunks:
        pred = {}
        cnt_index = 0
        is_invalid = False
        for se in se_order:
            if se == 'a' or se == 'o':
                start_index = el[cnt_index]
                end_index = el[cnt_index+1]
                cnt_index += 2

                try:
                    start_index = int(start_index)
                    end_index = int(end_index)
                    if end_index < start_index:
                        start_index, end_index = end_index, start_index
                    if start_index == -1 or end_index == -1:
                        pred[data_utils.SENTIMENT_ELEMENT[se]] = "NULL"
                    else:
                        word = splitted_text[start_index:end_index+1]
                        word = ' '.join(word)
                        pred[data_utils.SENTIMENT_ELEMENT[se]] = word
                except:
                    is_invalid = True
                    break
            elif se == 's':
                try:
                    sentiment = data_utils.SENTTAG2WORD[el[cnt_index]]
                    pred[data_utils.SENTIMENT_ELEMENT['s']] = sentiment
                except:
                    is_invalid = True
                    pass
                cnt_index += 1
            else: # c
                pred[data_utils.SENTIMENT_ELEMENT[se]] = el[cnt_index]
                cnt_index += 1
        if not is_invalid:
            result.append(pred)
    return result


# %%
ans = construct_answer(william_intermediate["hotel"]["oas"]["train"][0]["num_targets"])
text = william_intermediate["hotel"]["oas"]["train"][0]["text"]

# %%
william_intermediate["hotel"]["oas"]["train"][0]["target"]

# %%
catch_answer(ans,"oas",text)

# %% [markdown]
# # Data Preprocessing 2

# %%
from datasets import Dataset

william_2 = dict()
for domain, v1 in william_intermediate.items():
    william_2[domain] = {
        "train" : [], # basic task
        "val" : [], # complex task
        "test" : [] # complex task
    }
    # TRAIN
    for basic_task in task_tree["oas"]:
        for el in william_intermediate[domain][basic_task]["train"]:
            william_2[domain]["train"].append({
                    "input" : construct_prompt(el["text"],basic_task),
                    "output" : construct_answer(el["num_targets"]),
                    "task" : basic_task
                })
    # VAL
    for el in william_intermediate[domain]["oas"]["val"]:
        william_2[domain]["val"].append({
                "input" : construct_prompt(el["text"],"oas"),
                "output" : construct_answer(el["num_targets"]),
                "task" : "oas"
            })
    # TEST
    for el in william_intermediate[domain]["oas"]["test"]:
        william_2[domain]["test"].append({
                "input" : construct_prompt(el["text"],"oas"),
                "output" : construct_answer(el["num_targets"]),
                "task" : "oas"
            })
    william_2[domain]["train"] = Dataset.from_list(william_2[domain]["train"])
    william_2[domain]["val"] = Dataset.from_list(william_2[domain]["val"])
    william_2[domain]["test"] = Dataset.from_list(william_2[domain]["test"])

# %%
william_2["hotel"]["train"]["output"][2]

# %%
catch_answer(william_2["hotel"]["train"]["output"][2],william_2["hotel"]["train"]["task"][2],william_2["hotel"]["train"]["input"][2].split('|')[0])

# %% [markdown]
# # Prepare Tokenized Dataset

# %%
encoding_args = {
    "max_length" : 128,
    "padding" : True,
    "truncation" : True,
    "return_tensors" : "pt"
}

# %%
tokenizer_id = AutoTokenizer.from_pretrained("google/mt5-base")


# %%
def encode_id(dataset):
    result = tokenizer_id(dataset["input"], text_target=dataset["output"], **encoding_args)
    return result


# %%
william_tok = dict()
for domain, v1 in william_2.items():
    william_tok[domain] = dict()
    for split, v2 in v1.items():
        if split != "test":
            william_tok[domain][split] = william_2[domain][split].map(encode_id,batched=True,remove_columns=["input","output","task"])
            william_tok[domain][split].set_format("torch")
        else:
            william_tok[domain][split] = encode_id(william_2[domain][split])

# %% [markdown]
# # Data Collator

# %% [markdown]
# ## Indo

# %%
from transformers import DataCollatorForSeq2Seq

data_collator_id = DataCollatorForSeq2Seq(tokenizer=tokenizer_id)

# %% [markdown]
# # Compute Metrics

# %%
from transformers import EvalPrediction
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

def compute_metrics(eval_preds:EvalPrediction,decoding_args:Dict[str,str],tokenizer:AutoTokenizer,tasks:List) -> Dict[str,float]: # MAY NOT BE SUFFICIATE FOR CAUSAL LM
        """
        ### DESC
            Method to compute the metrics.
        ### PARAMS
        * eval_preds: EvalPrediction instance from training.
        * decoding_args: Decoding arguments.
        ### RETURN
        * metrics: Dictionary of metrics.
        """
        inputs, targets, predictions = preprocess_eval_preds(eval_preds,decoding_args,tokenizer)

        print("INPUTS >>",inputs[0])
        print("TARGETS >>",targets[0])
        print("PREDS >>",predictions[0])

        texts = [el.split('|')[0].strip() for el in inputs]

        targets = [catch_answer(ans,task,text) for ans,task,text in zip(targets,tasks,texts) if task != "non_absa"]
        predictions = [catch_answer(ans,task,text) for ans,task,text in zip(predictions,tasks,texts) if task != "non_absa"]

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


# %% [markdown]
# # Train Arguments

# %%
from transformers import Seq2SeqTrainingArguments

train_args = {
    "num_train_epochs": 10,
    "learning_rate": 3e-4,
    "save_total_limit": 2,
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": 8//n_gpu,
    "per_device_eval_batch_size": 8//n_gpu,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "logging_strategy" : "epoch",
    "metric_for_best_model": "overall_f1_score",
    "load_best_model_at_end": True,
    "adam_epsilon": 1e-08,
    "output_dir": "./output",
    "logging_dir" : "./output/log",
    "include_inputs_for_metrics" : True
}

train_args = Seq2SeqTrainingArguments(**train_args)

# %% [markdown]
# # Train

# %%
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %%
from transformers import Seq2SeqTrainer

# trainer = {
#     "peng" : {},
#     "wan" : {},
#     "zhang" : {},
#     "william" : {}
# }

decoding_args = {
    "skip_special_tokens" : False
}

def preprocess_logits_for_metrics(logits, targets):
    pred_logits = logits[0] if isinstance(logits,tuple) else logits
    pred_ids = torch.argmax(pred_logits, dim=-1)
    return pred_ids, targets


# %%
from tqdm import tqdm

def generate_predictions(model,tokenizer,data,device=torch.device("cuda:0"),decoding_args:Dict={}) -> List[str]:
    # Data loader
    # input_ids_data_loader = torch.utils.data.DataLoader(tokenized["input_ids"],
    #                     batch_size=batch_size,shuffle=False)
    # attention_mask_data_loader = torch.utils.data.DataLoader(tokenized["attention_mask"],
    #                     batch_size=batch_size,shuffle=False)
    # Predict
    model = model
    tokenizer = tokenizer
    tensor_predictions = []
    with torch.no_grad():
        for text in tqdm(data):
            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            tensor_predictions.extend(model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,max_length=128).cpu())
            input_ids = input_ids.cpu()
            # attention_mask = attention_mask.cpu()
    tensor_predictions = [[token for token in row if token != -100] for row in tensor_predictions]
    predictions = tokenizer.batch_decode(tensor_predictions,**decoding_args)
    predictions = [el for el in predictions]
    return predictions


# %%
import json

def save_result(str_preds_,preds,targets,filename):
    result = []
    str_preds = [el.replace("<pad>",'').replace("</s>",'') for el in str_preds_]
    assert len(str_preds) == len(preds) == len(targets)
    for i in range(len(str_preds)):
        result.append({
            "str_pred" : str_preds[i],
            "pred" : preds[i],
            "target" : targets[i]
        })
    
    with open(filename,'w') as fp:
        json.dump(result,fp)
    return result


# %% [markdown]
# # William Hotel

# %%
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model.to(device)
trainer = Seq2SeqTrainer(
        model = model,
        args = train_args,
        tokenizer = tokenizer_id,
        data_collator = data_collator_id,
        train_dataset = william_tok["hotel"]["train"],
        eval_dataset = william_tok["hotel"]["val"],
        compute_metrics = lambda eval_preds: compute_metrics(eval_preds,decoding_args,tokenizer_id,william_2["hotel"]["val"]["task"]),
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

trainer.train()

# %%
str_preds = generate_predictions(model, tokenizer_id, william_2["hotel"]["test"]["input"], device, decoding_args)
preds = [catch_answer(el,"oas") for el in str_preds]

# %%
targets = [catch_answer(el,"oas") for el in william_2["hotel"]["test"]["output"]]

# %%
summary_score(preds,targets)

# %%
# !rm -rf ./output

# %%
result = save_result(str_preds, preds, targets, "william_hotel.json")

# %%
result

# %%

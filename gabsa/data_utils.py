import json
import pandas as pd

from datasets import Dataset, DatasetDict

from preprocessing import process_numtargets

import random

def open_file(path):
    f = open(path,'r',encoding="utf-8")
    data = f.read()
    f.close()
    data = data.strip().split('\n')
    return data

available_task = ["ate","ote","aste","aope","uabsa"]

def read_files(paths,task):
    if task not in available_task:
        raise ValueError(f"Only insert available task. Available task : {available_task}")
    data = []
    for path in paths:
        data.extend(open_file(path))
    result = {
        "text" : [],
        "target" : [],
        "num_target" : []
    }
    for line in data:
        splitted_line = line.split('####')
        # Input
        text = splitted_line[0]
        # Output
        num_target = eval(splitted_line[1])
        target = process_numtargets(text,num_target,task=task)
        # Append
        result["text"].append(text)
        result["target"].append(target)
        result["num_target"].append(num_target)
    res = pd.DataFrame(result)
    return res

def build_gabsa_dataset(paths,tasks=["aste"],blank_frac=None,random_state=None,prompter=None,prompt_option=None,shuffle=True):
    data = pd.DataFrame()
    for t in tasks:
        data_ = read_files(paths,task=t)
        data_["task"] = t

        if prompter != None:
            data_["input"], data_["prompt"] = prompter.add_prompt(t,data_.text.values,prompt_option[t])
        else:
            data_["input"] = data_["text"]
            data_["prompt"] = ""
        
        data = pd.concat([data,data_])

    blank_frac = 1.0 if blank_frac == None else blank_frac
    if blank_frac < 1.0:
        len_label = data.target.apply(len)
        no_blank_data= data.loc[len_label > 0]

        if blank_frac == 0:
            data = no_blank_data
        else:
            blank_data = data.loc[len_label == 0].sample(frac=blank_frac,random_state=random_state)
            data = pd.concat([no_blank_data,blank_data])
    
    if shuffle:
        data = data.sample(frac=1,random_state=random_state).reset_index(drop=True)
    try:
        data["target"] = data.target.astype(str)
        data["num_target"] = data.num_target.astype(str)
        data = Dataset.from_pandas(data)
    except Exception as e:
        print(data.info())
        raise e
    
    return data

def build_gabsa_dataset_dict(train_paths,dev_paths,test_paths,do_train,do_eval,do_predict,task="aste",blank_frac=None,random_state=None,prompter=None,prompt_option_path=None,shuffle=True):
    tasks = task.split()
    prompt_option = json.load(open(prompt_option_path,'r',encoding="utf-8"))
    
    datasets = {}
    if do_train:
        datasets["train"] = build_gabsa_dataset(paths=train_paths,
        tasks=tasks,blank_frac=blank_frac,random_state=random_state,prompter=prompter,
        prompt_option=prompt_option,shuffle=shuffle)
    if do_eval:
        datasets["dev"] = build_gabsa_dataset(paths=dev_paths,
        tasks=tasks,blank_frac=blank_frac,random_state=random_state,prompter=prompter,
        prompt_option=prompt_option,shuffle=shuffle)
    if do_predict:
        datasets["test"] = build_gabsa_dataset(paths=test_paths,
        tasks=tasks,blank_frac=blank_frac,random_state=random_state,prompter=prompter,
        prompt_option=prompt_option,shuffle=shuffle)
    
    datasets = DatasetDict(datasets)

    if do_train:
        sample_train = datasets["train"][0]
        print("Sample train:",sample_train)
    if do_eval:
        sample_dev = datasets["dev"][0]
        print("Sample dev:",sample_dev)
    if do_predict:
        sample_test = datasets["test"][0]
        print("Sample test:",sample_test)

    return datasets
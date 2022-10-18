import json
import pandas as pd

from datasets import Dataset, DatasetDict

from preprocessing import process_numtargets

import numpy as np

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

def build_gabsa_dataset(train_paths,dev_paths,test_paths,task="aste",blank_frac=None,random_state=None,prompter=None,prompt_option_path=None,shuffle=True):
    tasks = task.split()
    train = pd.DataFrame()
    dev = pd.DataFrame()
    test = pd.DataFrame()
    prompt_option = json.load(open(prompt_option_path,'r',encoding="utf-8"))
    for t in tasks:
        train_ = read_files(train_paths,task=t)
        dev_ = read_files(dev_paths,task=t)
        test_ = read_files(test_paths,task=t)
        train_["task"] = t
        dev_["task"] = t
        test_["task"] = t

        if prompter != None:
            train_["input"], train_["prompt"] = prompter.add_prompt(t,train_.text.values,prompt_option[t])
            dev_["input"], dev_["prompt"] = prompter.add_prompt(t,dev_.text.values,prompt_option[t])
            test_["input"], test_["prompt"] = prompter.add_prompt(t,test_.text.values,prompt_option[t])
        else:
            train_["input"] = train_["text"]
            dev_["input"] = dev_["text"]
            test_["input"] = test_["text"]

            train_["prompt"] = ""
            dev_["prompt"] = ""
            test_["prompt"] = ""

        train = pd.concat([train,train_])
        dev = pd.concat([dev,dev_])
        test = pd.concat([test,test_])

    blank_frac = 1.0 if blank_frac == None else blank_frac
    if blank_frac < 1.0:
        len_label_train = train.target.apply(len)
        len_label_dev = dev.target.apply(len)
        len_label_test = test.target.apply(len)

        no_blank_train = train.loc[len_label_train > 0]
        no_blank_dev = dev.loc[len_label_dev > 0]
        no_blank_test = test.loc[len_label_test > 0]

        if blank_frac == 0:
            train = no_blank_train
            dev = no_blank_dev
            test = no_blank_test
        else:
            blank_train = train.loc[len_label_train == 0].sample(frac=blank_frac,random_state=random_state)
            blank_dev = dev.loc[len_label_dev == 0].sample(frac=blank_frac,random_state=random_state)
            blank_test = test.loc[len_label_test == 0].sample(frac=blank_frac,random_state=random_state)

            train = pd.concat([no_blank_train,blank_train])
            dev = pd.concat([no_blank_dev,blank_dev])
            test = pd.concat([no_blank_test,blank_test])
    
    if shuffle:
        train = train.sample(frac=1,random_state=random_state).reset_index(drop=True)
        dev = dev.sample(frac=1,random_state=random_state).reset_index(drop=True)
        test = test.sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    try:
        train["target"] = train.target.astype(str)
        dev["target"] = dev.target.astype(str)
        test["target"] = test.target.astype(str)

        train["num_target"] = train.num_target.astype(str)
        dev["num_target"] = dev.num_target.astype(str)
        test["num_target"] = test.num_target.astype(str)

        train = Dataset.from_pandas(train)
        dev = Dataset.from_pandas(dev)
        test = Dataset.from_pandas(test)
    except Exception as e:
        print(train.info())
        raise e
    
    # def batch_eval(x,colnames):
    #     result = {}
    #     for col in colnames:
    #         if col == "target" or col == "num_target":
    #             continue
    #         result[col] = x[col]
    #     result["target"] = [eval(el) for el in x["target"]]
    #     result["num_target"] = [eval(el) for el in x["num_target"]]
    #     return result
    
    # train = train.map(lambda x : batch_eval(x,train.column_names),batched=True)
    # dev = dev.map(lambda x : batch_eval(x,dev.column_names),batched=True)
    # test = test.map(lambda x : batch_eval(x,test.column_names),batched=True)
    # train["target"] = train["target"].map(eval)
    # dev["target"] = dev["target"].map(eval)
    # test["target"] = test["target"].map(eval)

    # train["num_target"] = train["num_target"].map(eval)
    # dev["num_target"] = dev["num_target"].map(eval)
    # test["num_target"] = test["num_target"].map(eval)

    sample_train = train[0]
    sample_dev = dev[0]
    sample_test = test[0]

    print("Sample train:",sample_train)
    print("Sample dev:",sample_dev)
    print("Sample test:",sample_test)

    datasets = DatasetDict({
        "train" : train,
        "dev" : dev,
        "test" : test
    })

    return datasets
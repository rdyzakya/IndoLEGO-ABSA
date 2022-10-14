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
        "target" : []
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
    res = pd.DataFrame(result)
    return res

def build_gabsa_dataset(train_paths,dev_paths,test_paths,task="aste",blank_frac=None,random_state=None,**kwargs):
    train = read_files(train_paths,task=task)
    dev = read_files(dev_paths,task=task)
    test = read_files(test_paths,task=task)

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
    train = Dataset.from_pandas(train)
    dev = Dataset.from_pandas(dev)
    test = Dataset.from_pandas(test)

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
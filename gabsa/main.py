import torch

import gpu

import argparse
import json
import os

import pandas as pd
import numpy as np

import logging

from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import AutoConfig, TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from datasets import load_metric
import model_types

from eval_utils import EvaluationCallback, compute_metrics, evaluate

from data_utils import build_gabsa_dataset
from model import get_gabsa_tokenizer_and_model
from preprocessing import batch_inverse_stringify_target, Prompter, Pattern, batch_stringify_target, post_process_lm_result, available_task, available_paradigm, no_target
from tqdm import tqdm
logger = logging.getLogger(__name__)

# Reference : https://github.com/huggingface/course/blob/main/chapters/en/chapter7/4.mdx
# Current issue training XGLM : https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783

def init_args():
    parser = argparse.ArgumentParser()
    # Training options
    parser.add_argument("--do_train",action="store_true",help="Do training phase")
    parser.add_argument("--do_eval",action="store_true",help="Do evaluation phase (validation)")
    parser.add_argument("--do_predict",action="store_true",help="Do testing phase (predict)")
    parser.add_argument("--train_args",type=str,help="Training argument (json)",required=False)
    # parser.add_argument("--n_gpu",type=str,help="GPU device this script will use",required=False)
    
    # Model options
    parser.add_argument("--model_type",type=str,help="Model type",required=False)
    parser.add_argument("--model_name_or_path",type=str,help="Model name or path",required=False)
    parser.add_argument("--max_len",type=int,help="Max sequence length for seq2seq model",default=128)
    
    # ABSA options
    parser.add_argument("--task",type=str,help="ABSA task",required=True)
    parser.add_argument("--paradigm",type=str,help="Paradigm (extraction,annotation)",default="extraction")
    parser.add_argument("--prompt_option_path",help="Prompt option path",required=False)
    parser.add_argument("--prompt_path",help="Path to prompt file (txt)",required=False)
    parser.add_argument("--pattern",help="Pattern file (json)",required=False)

    # Data arguments
    # parser.add_argument("--dataset_name",type=str,help="Dataset name for model folder naming convention", required=False)
    parser.add_argument("--data_dir",type=str,help="Dataset directory",default="")
    parser.add_argument("--trains",type=str,help="Training dataset",default="")
    parser.add_argument("--devs",type=str,help="Validation dataset",default="")
    parser.add_argument("--tests",type=str,help="Test dataset",default="")
    parser.add_argument("--blank_frac",type=float,help="Fraction of the blank label in the training dataset",default=1.0)
    
    # Misc
    parser.add_argument("--random_state",type=int,help="Random seed/state",default=42)
    parser.add_argument("--output_dir",type=str,help="Output directory",required=False)
    parser.add_argument("--per_device_predict_batch_size",type=int,help="Batch size per device for prediction phase",default=16)
    args = parser.parse_args()

    args.prompt_side = "left" if args.model_type in model_types.seq2seq else "right"
    if args.pattern != None:
        pattern_args = json.load(open(args.pattern,'r',encoding="utf-8"))
        args.pattern = Pattern(args.task,**pattern_args)
    return args

def batch_encode(x,args,tokenizer,encoding_args):
    # mask the targets
    masked_targets = []
    for target in x["target"]:
        target = eval(target)
        for i in range(len(target)):
            for k,v in target[i].items():
                target[i][k] = args.pattern.masking(v)
        masked_targets.append(str(target))
    text_target = batch_stringify_target(x["text"],x["num_target"],masked_targets,x["task"],args.paradigm,args.pattern)
    # mask the text input
    masked_inputs = []
    for el in x["input"]:
        masked_inputs.append(args.pattern.masking(el))
    if args.model_type in model_types.lm:
        inputs = [i + " " + t for i,t in zip(masked_inputs,text_target)]
        res = tokenizer(inputs, **encoding_args)
    elif args.model_type in model_types.seq2seq:
        res = tokenizer(masked_inputs, text_target=text_target, **encoding_args)
    else:
        raise NotImplementedError
    return res

def train_gabsa_model(args,dataset,prompter):
    tokenizer_args = {}
    model_args = {}
    tokenizer_args["padding_side"] = "left" if args.model_type in model_types.lm else "right"
    
    # Prepare the model and tokenizer
    model_and_tokenizer = get_gabsa_tokenizer_and_model(model_type=args.model_type,
    model_name_or_path=args.model_name_or_path,
    model_args=model_args,
    tokenizer_args=tokenizer_args)

    if model_and_tokenizer["tokenizer"].pad_token is None:
        model_and_tokenizer["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})

    add_new_terminology(model_and_tokenizer["tokenizer"],args.pattern,prompter)
    model_and_tokenizer["model"].resize_token_embeddings(len(model_and_tokenizer["tokenizer"]))
    
    # Prepare encoding arguments
    encoding_args = {
        "max_length" : args.max_len,
        "padding" : True,
        "truncation" : True,
        "return_tensors" : "pt"
    }

    decoding_args = {
        "skip_special_tokens" : True
    }

    # Set seed
    set_seed(args.random_state)

    # Load the training arguments
    args.train_args = args.train_args if args.train_args.endswith('.json') else args.train_args + ".json"
    train_args = json.load(open(args.train_args,'r'))

    # Prepare data collator
    if args.model_type not in model_types.seq2seq and args.model_type not in model_types.lm:
        raise ValueError("Model type is only from the seq2seq model and language modeling model (causal lm)")
    data_collator = DataCollatorForSeq2Seq(model_and_tokenizer["tokenizer"],model=model_and_tokenizer["model"]) \
        if args.model_type in model_types.seq2seq else DataCollatorForLanguageModeling(model_and_tokenizer["tokenizer"],mlm=False)
    

    evaluation_metrics_output_dir = os.path.join(args.output_dir,"eval_metrics.csv")
    cb = [EvaluationCallback(output_dir=evaluation_metrics_output_dir)] if args.do_eval else []
    
    # Setup training arguments (Trainarguments object)
    TrainingArgumentsClass = Seq2SeqTrainingArguments if args.model_type in model_types.seq2seq else TrainingArguments
    
    if args.model_type in model_types.seq2seq:
        train_args["predict_with_generate"] = True
    train_args.update({
        "output_dir" : args.output_dir,
        "logging_dir" : args.output_dir,
        "do_train" : args.do_train,
        "do_eval" : args.do_eval,
        "do_predict" : args.do_predict
    })

    train_args = TrainingArgumentsClass(**train_args)

    print("GPU used in training:",train_args.n_gpu)
    gpu.print_all_gpu_utilization()

    TrainerClass = Seq2SeqTrainer if args.model_type in model_types.seq2seq else Trainer
    tokenized_train = dataset["train"].map(
        lambda x : batch_encode(x=x,args=args,tokenizer=model_and_tokenizer["tokenizer"],encoding_args=encoding_args),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    trainer_args = {
        "model" : model_and_tokenizer["model"],
        "args" : train_args,
        "tokenizer" : model_and_tokenizer["tokenizer"],
        "data_collator" : data_collator,
        "train_dataset" : tokenized_train,
        "callbacks" : cb
    }

    if args.do_eval:

        def eval_metrics(eval_preds):
            dev_dataset = dataset["dev"]
            tokenizer = model_and_tokenizer["tokenizer"]

            return compute_metrics(eval_preds=eval_preds,
                                dataset=dev_dataset,
                                model_type=args.model_type,
                                paradigm=args.paradigm,
                                pattern=args.pattern,tokenizer=tokenizer,
                                encoding_args=encoding_args,decoding_args=decoding_args
                                )

        tokenized_dev = dataset["dev"].map(
                lambda x : batch_encode(x=x,args=args,tokenizer=model_and_tokenizer["tokenizer"],encoding_args=encoding_args),
                batched=True,
                remove_columns=dataset["dev"].column_names
            )

        trainer_args.update({
            "eval_dataset" : tokenized_dev,
            "compute_metrics" : eval_metrics,
        })

    trainer = TrainerClass(**trainer_args)

    # Train
    trainer.train()
    
    if trainer.is_world_process_zero():
        model_and_tokenizer["tokenizer"].save_pretrained(args.output_dir)
    model_and_tokenizer["model"].save_pretrained(save_directory=args.output_dir)

    print("Saving arguments...")
    temp_pattern = args.pattern
    args.pattern = args.pattern.pattern
    json.dump(vars(args),open(os.path.join(args.output_dir,"arguments.json"),'w',encoding="utf-8"))
    args.pattern = temp_pattern
    # for predicting
    args.model_name_or_path = args.output_dir

def predict_gabsa_model(args,dataset):
    # Prepare dataset
    test_dataset = dataset["test"]

    # Prepare model and tokenizer
    tokenizer_args = {}
    model_args = {}
    tokenizer_args["padding_side"] = "left" if args.model_type in model_types.lm else "right"
    
    # Prepare the model and tokenizer
    model_and_tokenizer = get_gabsa_tokenizer_and_model(model_type=args.model_type,
    model_name_or_path=args.model_name_or_path,
    model_args=model_args,
    tokenizer_args=tokenizer_args)

    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    # Prepare encoding arguments
    encoding_args = {
        "max_length" : args.max_len,
        "padding" : True,
        "truncation" : True,
        "return_tensors" : "pt"
    }

    decoding_args = {
        "skip_special_tokens" : True
    }

    # Setup device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tokenized_test = batch_encode(x=test_dataset,args=args,tokenizer=tokenizer,encoding_args=encoding_args)
    # Generate the predictions
    model.to(device)
    # tokenized_test = tokenized_test.to(device)

    # Data loader
    data_loader = torch.utils.data.DataLoader(tokenized_test["input_ids"],
                        batch_size=args.per_device_predict_batch_size,shuffle=False)
    
    # Predict
    tensor_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            tensor_preds.extend(model.generate(input_ids=batch.to(device),max_length=args.max_len))
    decoded_preds = tokenizer.batch_decode(tensor_preds,**decoding_args)
    
    if args.model_type in model_types.lm:
        decoded_preds = post_process_lm_result(texts_without_target=test_dataset["input"],
        texts_with_target=decoded_preds,tokenizer=tokenizer,encoding_args=encoding_args,
        decoding_args=decoding_args)
    
    inverse_stringified_preds = batch_inverse_stringify_target(batch_stringified_target=decoded_preds,
                                        batch_task=test_dataset["task"],
                                        paradigm=args.paradigm,
                                        pattern=args.pattern)
    
    # Calculate metrics
    evaluation_metrics = evaluate(pred_pt=inverse_stringified_preds,gold_pt=[eval(el) for el in test_dataset["target"]])
    print(evaluation_metrics)
    # Save dataset and metrics
    result_dataset = pd.DataFrame({
        "text" : test_dataset["text"],
        "prompt" : test_dataset["prompt"],
        "target" : test_dataset["target"],
        "prediction" : inverse_stringified_preds,
        "string_preds" : decoded_preds
    })

    # save prediction dataset
    result_dataset.to_csv(os.path.join(args.output_dir,"prediction.csv"),index=False)
    # save metric
    json.dump(evaluation_metrics,open(os.path.join(args.output_dir,"prediction_metrics.json"),'w',encoding="utf-8"))

def add_new_terminology(tokenizer,pattern,prompter):
    terms = available_task + available_paradigm + list(pattern.mask.keys())  \
        + list(pattern.mask.values()) # pattern characters
    all_prompts = []
    for task in prompter.prompts.keys():
        prompts = " ".join(prompter.prompts[task])
        all_prompts.extend(prompts.split())
    prompter_term = list(set(all_prompts))
    terms.extend(prompter_term)
    for t in terms:
        if t != tokenizer.convert_tokens_to_string(tokenizer.tokenize(t)):
            tokenizer.add_tokens(t)
    

def main():
    # Setup the arguments of the script
    args = init_args()
    # LOL
    if not args.do_train and not args.do_eval and not args.do_predict:
        logger.info("do_train is False, do_eval is False, do_eval_best is False, do_predict is False too. WHAT DO YOU WANT TO DO THEN???")
        return

    # Load the dataset
    # Prepare the dataset paths
    train_paths, dev_paths, test_paths = [], [], []
    for fname in args.trains.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        train_paths.append(os.path.join(args.data_dir,fname))
    for fname in args.devs.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        dev_paths.append(os.path.join(args.data_dir,fname))
    for fname in args.tests.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        test_paths.append(os.path.join(args.data_dir,fname))
    
    prompter = Prompter(args.prompt_path,args.prompt_side) if args.prompt_path != None else None
    
    dataset = build_gabsa_dataset(train_paths=train_paths,
                                dev_paths=dev_paths,
                                test_paths=test_paths,
                                task = args.task,
                                blank_frac=args.blank_frac,
                                random_state=args.random_state,
                                prompter=prompter,
                                prompt_option_path=args.prompt_option_path)
    
    # # Prepare output dir
    # model_size = "nosize"
    # available_size = ["small","large","base"]
    # for size in available_size:
    #     if size in args.model_name_or_path:
    #         model_size = size
    
    # base_output_dir = args.output_dir
    
    if args.do_train:
        tasks = args.task.split()
        task_name = "pabsa" if len(tasks) > 1 else tasks[0]
        # dataset_name = args.dataset_name or os.path.split(args.data_dir)[-1]
        pd = model_types.pd[args.model_type][args.model_name_or_path]
        output_dir = os.path.join(args.output_dir,args.model_type,f"{args.model_type}_{task_name}_S{args.max_len}_{pd}_blank={args.blank_frac}")
        if not os.path.exists(output_dir):
            os.makedirs(os.path.join(output_dir,"data"))
        
        args.output_dir = output_dir

        dataset["train"].to_csv(os.path.join(args.output_dir,"data","train.csv"),index=False)
        dataset["dev"].to_csv(os.path.join(args.output_dir,"data","dev.csv"),index=False)
        train_gabsa_model(args=args,dataset=dataset,prompter=prompter)
    if args.do_predict:
        data_dir = os.path.join(args.output_dir,"data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        dataset["test"].to_csv(os.path.join(data_dir,"test.csv"),index=False)
        # args.output_dir = base_output_dir
        predict_gabsa_model(args=args,dataset=dataset)

if __name__ == "__main__":
    main()
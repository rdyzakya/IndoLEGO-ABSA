import argparse
from gc import callbacks
import json
import os

import pandas as pd
import numpy as np

import logging

from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import AutoConfig, TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, set_seed
from datasets import load_metric

from eval_utils import EvaluationCallback, aste_compute_metrics, triplet_detection_compute_metrics

import torch

from data_utils import build_gaste_dataset, build_triplet_detector_dataset
from model import get_gaste_tokenizer_and_model, get_triplet_existence_model_and_tokenizer

from preprocessing import batch_inverse_stringify_target
logger = logging.getLogger(__name__)

# seq2seq = ["t5","byt5","mt5"]
# lm = ["xglm"]
import model_types

# Reference : https://github.com/huggingface/course/blob/main/chapters/en/chapter7/4.mdx
# Current issue training XGLM : https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",action="store_true",help="Do training phase")
    parser.add_argument("--do_eval",action="store_true",help="Do evaluation phase (validation)")
    parser.add_argument("--do_eval_best",action="store_true",help="Do evaluation phase with only the best checkpoint")
    parser.add_argument("--do_predict",action="store_true",help="Do testing phase (predict)")

    parser.add_argument("--n_gpu",type=int,help="GPU device this script will use",required=False)
    parser.add_argument("--output_dir",type=str,help="Output directory",required=False)

    parser.add_argument("--data_dir",type=str,help="Dataset directory",required=False)
    parser.add_argument("--aste_train_args",type=str,help="ASTE Training argument (json)",required=False)
    parser.add_argument("--triplet_detection_train_args",type=str,help="Triplet detection Training argument (json)",required=False)
    parser.add_argument("--train",type=str,help="Training dataset",required=False)
    parser.add_argument("--dev",type=str,help="Validation dataset",required=False)
    parser.add_argument("--test",type=str,help="Test dataset",required=False)

    parser.add_argument("--aste_model_type",type=str,help="ASTE model type",required=False)
    parser.add_argument("--aste_model_name_or_path",type=str,help="ASTE model name or path",required=False)
    parser.add_argument("--triplet_detection_model_type",type=str,help="Triplet detection model type",required=False)
    parser.add_argument("--triplet_detection_model_name_or_path",type=str,help="Triplet detection model name or path",required=False)

    parser.add_argument("--blank_frac",type=float,help="Fraction of the blank label in the training dataset",default=1.0)
    parser.add_argument("--random_state",type=int,help="Random seed/state",default=None)
    parser.add_argument("--aste_model",action="store_true",help="Training or validating or predicting using aste model")
    parser.add_argument("--triplet_detection_model",action="store_true",help="Training or validating or predicting using triplet detection model")

    parser.add_argument("--max_len",type=int,help="Max sequence length for seq2seq model",default=128)
    
    # for lm
    parser.add_argument("--prompt_option",help="Prompt option for causal LM model type",default=0)
    parser.add_argument("--quote",help="Quote the text for causal LM model type",action="store_true")
    parser.add_argument("--quote_with_space",help="Quote with space the text for causal LM model type",action="store_true")
    
    args = parser.parse_args()

    if args.prompt_option != "random":
        args.prompt_option = int(args.prompt_option)

    return args

def train_aste_model(args,device,train_paths,dev_paths,test_paths):
    # Prepare the model and tokenizer
    model_and_tokenizer = get_gaste_tokenizer_and_model(args.aste_model_type,args.aste_model_name_or_path)
    
    # Prepare tokenizer arguments
    tokenizer_args = {
        "max_length" : args.max_len,
        "padding" : True,
        "truncation" : True,
        "return_tensors" : "pt"
    }

    # Load the dataset
    dataset, tokenized_dataset = build_gaste_dataset(args.aste_model_type,
        model_and_tokenizer["tokenizer"],
        train_paths,dev_paths,
        test_paths,args.blank_frac,
        args.random_state,
        **tokenizer_args)

    # Set seed
    set_seed(args.random_state)

    # move model to gpu
    model_and_tokenizer["model"].to(device)

    if args.do_train:
        # Load the training arguments
        args.aste_train_args = args.aste_train_args if args.aste_train_args.endswith('.json') else args.aste_train_args + ".json"
        aste_train_args = json.load(open(args.aste_train_args,'r'))

        # Prepare data collator
        if args.aste_model_type not in model_types.seq2seq and args.aste_model_type not in model_types.lm:
            raise ValueError("Model type is only from the seq2seq model and language modeling model (causal lm)")
        data_collator = DataCollatorForSeq2Seq(model_and_tokenizer["tokenizer"],model=model_and_tokenizer["model"]) \
            if args.aste_model_type in model_types.seq2seq else DataCollatorForLanguageModeling(model_and_tokenizer["tokenizer"],mlm=False)
        
        # Prepare output dir and logging dir
        model_size = "nosize"
        available_size = ["small","large","base"]
        for size in available_size:
            if size in args.aste_model_name_or_path:
                model_size = size

        output_dir = os.path.join(args.output_dir,args.aste_model_type,f"{args.aste_model_type}_aste_{args.max_len}_{model_size}_blank={args.blank_frac}")
        # logging_dir = os.path.join(args.output_dir,args.model_type,"logs",f"{args.aste_model_type}_aste_{args.max_len}_blank={args.blank_frac}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        evaluation_metrics_output_dir = os.path.join(output_dir,"eval_metrics.csv")
        cb = [EvaluationCallback(output_dir=evaluation_metrics_output_dir)] if args.do_eval else []
        
        # Setup training arguments (Trainarguments object)
        TrainingArgumentsClass = Seq2SeqTrainingArguments if args.aste_model_type in model_types.seq2seq else TrainingArguments
        
        if args.aste_model_type in model_types.seq2seq:
            aste_train_args["predict_with_generate"] = True

        aste_train_args = TrainingArgumentsClass(
            output_dir=output_dir,
            logging_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            **aste_train_args
        )

        print("GPU used by aste training:",aste_train_args.n_gpu)

        TrainerClass = Seq2SeqTrainer if args.aste_model_type in model_types.seq2seq else Trainer

        trainer = TrainerClass(
            model=model_and_tokenizer["model"],
            args=aste_train_args,
            tokenizer=model_and_tokenizer["tokenizer"],
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            compute_metrics=lambda x : aste_compute_metrics(x,dataset["dev"],model_and_tokenizer["tokenizer"],
                                        model_type=args.aste_model_type,prompt_option=args.prompt_option,
                                        quote=args.quote,quote_with_space=args.quote_with_space,**tokenizer_args),
            callbacks=cb
        )

        # Train
        trainer.train()

        print("Saving arguments...")

        json.dump(vars(args),open(os.path.join(output_dir,"arguments.json"),"w",encoding="utf-8"))

        if trainer.is_world_process_zero():
            model_and_tokenizer["tokenizer"].save_pretrained(output_dir)
        model_and_tokenizer["model"].save_pretrained(save_directory=output_dir)

        if args.do_predict:
            test_dataset = dataset["test"]
            tokenized_test_dataset = tokenized_dataset["test"]

            predictions = trainer.predict(tokenized_test_dataset)
            predictions_token = predictions.predictions
            try:
                labels_token = predictions.label_ids
            except:
                print("Error label ids!")
                print(dir(predictions))
                exit()
            inputs_token = tokenized_test_dataset["input_ids"]

            decoded_input = model_and_tokenizer["tokenizer"].batch_decode(inputs_token, skip_special_tokens=True)
            decoded_preds = model_and_tokenizer["tokenizer"].batch_decode(predictions_token, skip_special_tokens=True)
            decoded_labels = model_and_tokenizer["tokenizer"].batch_decode(labels_token, skip_special_tokens=True)
            
            inverse_stringified_preds = batch_inverse_stringify_target(decoded_preds)
            inverse_stringified_labels = batch_inverse_stringify_target(decoded_labels)
            real_labels = test_dataset["target"]

            try:
                metrics = pd.DataFrame([predictions.metrics])
            except:
                print("Error metrics!!!!")
                print(predictions.metrics)
                exit()

            df_test_prediction = pd.DataFrame({
                "text" : test_dataset["text"],
                "tokenized_text" : decoded_input,
                "prediction" : inverse_stringified_preds,
                "target" : real_labels,
                "tokenized_target" : inverse_stringified_labels
            })

            df_test_prediction_output_dir = os.path.join(output_dir,"prediction.csv")
            df_test_metrics_output_dir = os.path.join(output_dir,"prediction_metrics.csv")

            df_test_prediction.to_csv(df_test_prediction_output_dir)
            metrics.to_csv(df_test_metrics_output_dir)

def train_triplet_detection_model(args,device,train_paths,dev_paths,test_paths):
    # Prepare the model and tokenizer
    model_and_tokenizer = get_triplet_existence_model_and_tokenizer(args.triplet_detection_model_type,args.triplet_detection_model_name_or_path)
    
    # Prepare tokenizer arguments
    tokenizer_args = {
        "max_length" : args.max_len,
        "padding" : True,
        "truncation" : True,
        "return_tensors" : "pt"
    }

    # Load the dataset
    dataset, tokenized_dataset = build_triplet_detector_dataset(
        model_and_tokenizer["tokenizer"],
        train_paths,dev_paths,
        test_paths,
        **tokenizer_args)

    # Set seed
    set_seed(args.random_state)

    # move model to gpu
    model_and_tokenizer["model"].to(device)

    if args.do_train:
        # Load the training arguments
        args.triplet_detection_train_args = args.triplet_detection_train_args if args.triplet_detection_train_args.endswith('.json') else args.triplet_detection_train_args + ".json"
        triplet_detection_train_args = json.load(open(args.triplet_detection_train_args,'r'))

        # Prepare data collator
        data_collator = DataCollatorWithPadding(model_and_tokenizer["tokenizer"])
        
        # Prepare output dir and logging dir
        model_size = "nosize"
        available_size = ["small","large","base"]
        for size in available_size:
            if size in args.triplet_detection_model_name_or_path:
                model_size = size

        output_dir = os.path.join(args.output_dir,args.triplet_detection_model_type,f"{args.triplet_detection_model_type}_triplet_detection_{args.max_len}_{model_size}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Setup training arguments (Trainarguments object)
        triplet_detection_train_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            **triplet_detection_train_args
        )


        trainer = Trainer(
            model=model_and_tokenizer["model"],
            args=triplet_detection_train_args,
            tokenizer=model_and_tokenizer["tokenizer"],
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            compute_metrics=triplet_detection_compute_metrics
        )

        # Train
        trainer.train()

        print("Saving arguments...")

        json.dump(vars(args),open(os.path.join(output_dir,"arguments.json"),"w",encoding="utf-8"))

        if trainer.is_world_process_zero():
            model_and_tokenizer["tokenizer"].save_pretrained(output_dir)
        model_and_tokenizer["model"].save_pretrained(save_directory=output_dir)

        if args.do_predict:
            test_dataset = dataset["test"]
            tokenized_test_dataset = tokenized_dataset["test"]
            predictions = trainer.predict(tokenized_test_dataset).predictions
            predictions = np.argmax(predictions,axis=-1)

            test_result = pd.DataFrame({
                "text" : test_dataset["text"],
                "label" : test_dataset["label"],
                "preds" : predictions
            })

            metrics = pd.DataFrame([predictions.metrics])

            df_test_prediction_output_dir = os.path.join(output_dir,"prediction.csv")
            df_test_metrics_output_dir = os.path.join(output_dir,"prediction_metrics.csv")

            test_result.to_csv(df_test_prediction_output_dir)
            metrics.to_csv(df_test_metrics_output_dir)



def main():
    # Setup the arguments of the script
    args = init_args()
    # LOL
    if not args.do_train and not args.do_eval and not args.do_predict:
        logger.info("do_train is False, do_eval is False, do_eval_best is False, do_predict is False too. WHAT DO YOU WANT TO DO THEN???")
        return
    
    # Setup device
    # os.environ['CUDA_VISIBLE_DEVICES']= str(args.n_gpu)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device : {device}")
    
    # Prepare the dataset paths
    train_paths, dev_paths, test_paths = [], [], []
    for fname in args.train.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        train_paths.append(os.path.join(args.data_dir,fname))
    for fname in args.dev.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        dev_paths.append(os.path.join(args.data_dir,fname))
    for fname in args.test.split():
        if not fname.endswith(".txt"):
            fname += ".txt"
        test_paths.append(os.path.join(args.data_dir,fname))
    
    if args.aste_model:
        train_aste_model(args,device,train_paths,dev_paths,test_paths)
    if args.triplet_detection_model:
        train_triplet_detection_model(args,device,train_paths,dev_paths,test_paths)

main()
from dataset import ABSADataset, NonABSADataset, MixedDataset, Pattern, Prompter
from model import ABSAGenerativeModelWrapper
from training import ABSAGenerativeTrainer

from typing import Dict, Any

from argparse import ArgumentParser

import json

import os

def init_args() -> Dict[str,Any]:
    parser = ArgumentParser()

    # All arguments
    parser.add_argument("--args_json",type=str,help="Arguments file (json)",required=False)

    # Train arguments
    parser.add_argument("--cuda_device",type=str,help="GPU device use for training",default="0")
    parser.add_argument("--training_args",type=str,help="Training arguments path (json)",default="train_args.json")

    # Model options
    parser.add_argument("--model_and_tokenizer_args",type=str,help="Model and tokenizer args",required=False)
    parser.add_argument("--model_name_or_path",type=str,help="Model name or path",default="t5-base")

    # Data
    parser.add_argument("--data_config",type=str,help="Path to data configuration (json)",default="data_config.json")

    # Pattern options
    parser.add_argument("--pattern_config",type=str,help="Path to pattern object configuration",required=False)

    # Prompt options
    parser.add_argument("--prompt_config",type=str,help="Path to prompter object configuration",required=False)

    # Misc
    parser.add_argument("--output_dir",type=str,help="Directory for output",default="./output")
    parser.add_argument("--trainer_seed",type=int,help="Trainer seed",default=None)

    args = parser.parse_args()
    # Transform argparse.NameSpace --> Dict
    args = vars(args)

    if "args_json" in args.keys():
        with open(args["args_json"],'r') as file_object:
            loaded_args = json.load(file_object)
        args.update(loaded_args)
    return args


def main():
    # Arguments
    args = init_args()
    
    # Prepare the model
    model_and_tokenizer_args = args["model_and_tokenizer_args"] if "model_and_tokenizer_args" in args.keys() else {}
    model_and_tokenizer = ABSAGenerativeModelWrapper(args["model_name_or_path"],**model_and_tokenizer_args)
    
    # Prepare the data
    with open(args["data_config"],'r') as file_object:
        data_config = json.load(file_object)
    
    pattern_object = Pattern() if "pattern_config" not in args.keys() else Pattern(**args["pattern_config"])
    prompter = Prompter() if "prompt_config" not in args.keys() else Prompter(**args["prompt_config"])

    datasets = {
        "train" : [],
        "val" : [],
        "test" : []
    }

    for data_split in datasets.keys():
        ## ABSA Dataset
        for data_config in data_config["absa"][data_split]:
            datasets[data_split].append(ABSADataset(prompter=prompter,
                                            prompt_side=model_and_tokenizer.prompt_side,
                                            pattern=pattern_object,
                                            **data_config))
        ## NonABSA Dataset
        for data_config in data_config["non_absa"][data_split]:
            datasets[data_split].append(NonABSADataset(prompt_side=model_and_tokenizer.prompt_side,
                                                **data_config))
    ## Transform to MixedDataset
    train_dataset = MixedDataset(datasets["train"],
                                shuffle=data_config["shuffle"]["train"],
                                seed=data_config["seed"]["train"])
    val_dataset = MixedDataset(datasets["val"],
                                 shuffle=data_config["shuffle"]["val"],
                                 seed=data_config["seed"]["val"]) if args["training_args"]["do_eval"] else None
    test_dataset = MixedDataset(datasets["test"],
                                 shuffle=data_config["shuffle"]["test"],
                                 seed=data_config["seed"]["test"]) if args["training_args"]["do_predict"] else None
    
    # Train the model
    encoding_args = {}
    trainer = ABSAGenerativeTrainer(model_and_tokenizer,pattern=pattern_object)
    trainer.prepare_data(train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         test_dataset=test_dataset,
                         **encoding_args)
    trainer.compile_train_args(args["training_args"])
    trainer.prepare_trainer()

    if args["training_args"]["do_train"]:
        trainer.train(output_dir=args["output_dir"],
                    random_seed=args["trainer_seed"])
    
    # Prediction
    if args["training_args"]["do_predict"]:
        trainer.predict()

if __name__ == "__main__":
    main()
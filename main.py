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
    parser.add_argument("--model_name_or_path",type=str,help="Model name or path",default="t5-base")

    # Data
    parser.add_argument("--data_config",type=str,help="Path to data configuration (json)",default="data_config.json")
    # parser.add_argument("--absa_data_dir",type=str,help="ABSA dataset directory",default="./absa_data")
    # parser.add_argument("--absa_train_data",nargs='*',type=str,help="List of absa train data file names (txt format)",default=["train.txt"])
    # parser.add_argument("--absa_val_data',nargs='*",type=str,help="List of absa validation data file names (txt format)",default=["val.txt"])
    # parser.add_argument("--absa_test_data",nargs='*',type=str,help="List of absa test data file names (txt format)",default=["test.txt"])

    # parser.add_argument("--absa_train_target_format",nargs='*',type=str,help="List of absa train target format (combination of a [aspect] c [category] o [opinion] s [sentiment])",default=["acos"])
    # parser.add_argument("--absa_val_target_format",nargs='*',type=str,help="List of absa val target format (combination of a [aspect] c [category] o [opinion] s [sentiment])",default=["acos"])
    # parser.add_argument("--absa_test_target_format",nargs='*',type=str,help="List of absa test target format (combination of a [aspect] c [category] o [opinion] s [sentiment])",default=["acos"])

    # parser.add_argument("--non_absa_data_dir",type=str,help="NonABSA dataset directory",default="./non_absa_data")
    # parser.add_argument("--non_absa_train_data",nargs='*',type=str,help="List of non absa train data file names (csv format)",default=["train.csv"])
    # parser.add_argument("--non_absa_val_data",nargs='*',type=str,help="List of non absa validation data file names (csv format)",default=["val.csv"])
    # parser.add_argument("--non_absa_test_data",nargs='*',type=str,help="List of non absa test data file names (csv format)",default=["test.csv"])

    # Pattern options
    parser.add_argument("--pattern_open_bracket",type=str,help="Pattern open bracket",default='(')
    parser.add_argument("--pattern_close_bracket",type=str,help="Pattern close bracket",default=')')
    parser.add_argument("--pattern_intra_sep",type=str,help="Pattern intra-seperator",default=',')
    parser.add_argument("--pattern_inter_sep",type=str,help="Pattern inter-seperator",default=';')

    # Prompt options
    parser.add_argument("--extraction_prompt",type=str,help="Prompt for the extraction paradigm",required=False)
    parser.add_argument("--imputation_prompt",type=str,help="Prompt for the imputation paradigm",required=False)

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
    # Prepare the data
    # Train the model
    # Prediction

if __name__ == "__main__":
    main()
import argparse
import time

import torch
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import re
import sys

sys.path.append("/srv/nas_data1/text/randy/aste/facebook-aste/gaste")

from eval_utils import evaluate
from data_utils import read_files
from fixing_utils import fix_preds_aste
from preprocessing import postprocess_gaste_output, batch_preprocess_aste
from model import get_gaste_tokenizer_and_model, get_triplet_existence_model_and_tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--aste_model_type", type=str, required=True, help="ASTE model type")
parser.add_argument("--aste_model_name_or_path", type=str, required=True, help="ASTE model name or path")
parser.add_argument("--triplet_detection_model_type",type=str,required=False,help="Triplet detection model type")
parser.add_argument("--triplet_detection_model_name_or_path",type=str,required=False,help="Triplet detection model name or path")
parser.add_argument("--data", type=str, required=True,
                        help="Test data")
parser.add_argument("--n_gpu", type=str, required=True,
                        help="GPU used")
parser.add_argument("--fixing", type=str, help="Fixing option", choices=["raw","editdistance","cut","remove"],default="raw")
parser.add_argument("--output_dir",type=str,help="Output directory",required=True)
parser.add_argument("--batch_size",type=int,help="Testing batch size",default=16)
parser.add_argument("--triplet_detection",action="store_true",help="Wether to detect triplet or not")

args = parser.parse_args()

# Data prep
dataset = read_files([args.data],"aste")

# Setup device
device = torch.device(f'cuda:{args.n_gpu}')

aste_input = dataset["text"]
aste_target = dataset["target"]

# Model preparation
aste_model_and_tokenizer = get_gaste_tokenizer_and_model(args.aste_model_type,args.aste_model_name_or_path)
aste_model = aste_model_and_tokenizer["model"]
aste_tokenizer = aste_model_and_tokenizer["tokenizer"]
aste_model.to(device)

if args.triplet_detection:
    triplet_detection_model_and_tokenizer = get_triplet_existence_model_and_tokenizer(args.triplet_detection_model_type,
    args.triplet_detection_model_name_or_path)
    triplet_detection_model = triplet_detection_model_and_tokenizer["model"]
    triplet_detection_tokenizer = triplet_detection_model_and_tokenizer["tokenizer"]
    triplet_detection_model.to(device)


global_start_time = time.time()

if args.triplet_detection:
    # Do triplet detection
    triplet_detection_tokenized_input = triplet_detection_tokenizer.batch_encode_plus(dataset["text"].tolist(), max_length=triplet_detection_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
    triplet_detection_data_loader = torch.utils.data.DataLoader(triplet_detection_tokenized_input["input_ids"], batch_size=args.batch_size,shuffle=False)
    triplet_detection_start_time = time.time()
    triplet_detection_result = []
    triplet_detection_model_start_time = time.time()
    for batch in tqdm(triplet_detection_data_loader):
        triplet_detection_result.extend(triplet_detection_model(input_ids=batch))
    triplet_detection_model_end_time = time.time()
    triplet_detection_result = np.argmax(triplet_detection_result,axis=-1)
    dataset["exist_triplet"] = triplet_detection_result
    aste_input = dataset.loc[dataset["exist_triplet"] == 1,"text"]
    aste_target = dataset.loc[dataset["exist_triplet"] == 1,"target"]

    print("Result triplet detection:",aste_input.exist_triplet.value_counts())

# batch_stringified_target = batch_stringify_target(aste_target.tolist())
# assert len(aste_input) == len(batch_stringified_target)

aste_dataset = {
    "text" : aste_input,
    "target" : aste_target
}

tokenizer_args = {
        "max_length" : aste_model.config.max_length,
        "padding" : True,
        "truncation" : True,
        "return_tensors" : "pt"
    }

print("First 5 data :")
print(aste_dataset["text"][:5])

aste_tokenized_input = batch_preprocess_aste(aste_tokenizer,aste_dataset,args.aste_model_type,**tokenizer_args).to(device)

# Tokenize
# aste_tokenized_input = aste_tokenizer(aste_input.tolist(), text_target=batch_stringified_target, max_length=aste_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)

# Prep data loader
aste_data_loader = torch.utils.data.DataLoader(aste_tokenized_input["input_ids"],batch_size=args.batch_size,shuffle=False)

print("Generating results...")
result_ = []
aste_model_start_time = time.time()
for batch in tqdm(aste_data_loader):
    result_.extend(aste_model.generate(input_ids=batch).to('cpu'))
aste_model_end_time = time.time()

# Decode and detokenize
result = aste_tokenizer.batch_decode(result_, skip_special_tokens=True, **tokenizer_args)
print("Few result (5 first results):")
print(result[:5])
# Inverse stringify
result = postprocess_gaste_output(dataset["text"],result,args.aste_model_type,aste_tokenizer,
prompt_option=0,quote=True,quote_with_space=True,**tokenizer_args)


if args.fixing:
    if args.fixing != "raw":
        result = fix_preds_aste(result,dataset["text"],option=args.fixing)

global_end_time = time.time()

if args.triplet_detection:
    dataset["preds"] = [[] for _ in range(dataset.shape[0])]
    dataset.loc[dataset["exist_triplet"] == 1,"preds"] = result
else:
    dataset["preds"] = result

# Calculate metrics
print("Calculation evaluation...")
evaluation = evaluate(dataset["text"],dataset["target"],dataset["preds"])
print(evaluation)

model_info = {
    "aste_model_type" : args.aste_model_type,
    "aste_model_name_or_path" : args.aste_model_name_or_path,
    "fixing" : args.fixing
}

inference_time = {
    "global" : global_end_time - global_start_time,
    "aste" : aste_model_end_time - aste_model_start_time
}

if args.triplet_detection:
    model_info["triplet_detection_model_type"] = args.triplet_detection_model_type
    model_info["triplet_detection_model_name_or_path"] = args.triplet_detection_model_name_or_path
    inference_time["triplet_detection"] = triplet_detection_model_end_time - triplet_detection_model_end_time

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

json.dump(evaluation,open(os.path.join(args.output_dir,"evaluation.json"),"w"))
json.dump(model_info,open(os.path.join(args.output_dir,"model_info.json"),"w"))
json.dump(inference_time,open(os.path.join(args.output_dir,"inference_time.json"),"w"))
dataset.to_csv(os.path.join(args.output_dir,"prediction.csv"),index=None)
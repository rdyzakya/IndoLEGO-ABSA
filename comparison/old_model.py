import argparse
import time
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import numpy as np
import nltk

import sys
sys.path.append("/srv/nas_data1/text/randy/absa/facebook-absa/gabsa")

from data_utils import read_files
from eval_utils import evaluate

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}

def catch_token_span(token_list,polarized=True):
    result_index = []
    current_token = []
    for i in range(len(token_list)):
        if token_list[i] == 'O':
            if len(current_token) > 0:
                result_index.append(current_token)
            current_token = []
            continue
        if token_list[i][0] == 'B' and len(current_token) == 0:
            current_token.append(i)
        elif token_list[i][0] == 'B' and len(current_token) > 0:
            result_index.append(current_token)
            current_token = [i]
        elif token_list[i][0] == 'I' and len(current_token) > 0 and token_list[i][2:] == token_list[current_token[-1]][2:]:
            current_token.append(i)
        elif token_list[i][0] == 'I' and len(current_token) > 0 and token_list[i][2:] != token_list[current_token[-1]][2:]:
            result_index.append(current_token)
            current_token = []
        elif token_list[i][0] == 'I' and len(current_token) == 0:
            current_token = []
    if polarized:
        return {
            'POS' : [el for el in result_index if token_list[el[0]][2:] == 'POS'],
            'NEG' : [el for el in result_index if token_list[el[0]][2:] == 'NEG']
        }
    # else
    return result_index

special_tokens = ["[PAD]","[SEP]","[UNK]","[CLS]","[MASK]"]

def detokenize(tokenizer,tokens,indexs,**kwargs):
    result = []
    try:
        for i in indexs:
            if tokens[i] not in special_tokens:
                result.append(tokens[i])
    except Exception as e:
        print("Tokens:",tokens)
        print("Indexs:",indexs)
        raise e
    return tokenizer.convert_tokens_to_string(result,**kwargs)

def postprocess_bio(tokens):
    # BIO tokens (can be used to non polarized BIO)
    result = []
    for i in range(len(tokens)):
        if i == 0:
            if tokens[i][0] == 'I':
                result.append(f"B-{tokens[i][1:]}")
            else:
                result.append(tokens[i])
        else:
            if tokens[i][0] == 'I':
                if tokens[i-1] == 'O' or (tokens[i][1:] != tokens[i-1][1:]):
                    result.append(f"B-{tokens[i][1:]}")
                else:
                    result.append(tokens[i])
            else:
                result.append(tokens[i])
    return result

def fixing(sent,term, is_subword=True):
    splitted_sent = sent.split()
    splitted_term = term.split()
    min_edit_distance = np.inf
    chosen_term = None
    for i in range(0,len(splitted_sent)-len(splitted_term) + 1):
        window = splitted_sent[i:i+len(splitted_term)]
        joined_window = ' '.join(window)
        edit_distance = nltk.edit_distance(joined_window,term)
        condition = edit_distance < min_edit_distance
        condition = condition and term in joined_window if is_subword else condition
        if condition:
            min_edit_distance = edit_distance
            chosen_term = joined_window
    return chosen_term
    

parser = argparse.ArgumentParser()
parser.add_argument("--concept", type=str, required=True,
                        help="Concept model checkpoint")
parser.add_argument("--sm", type=str, required=True,
                        help="Sentiment marker model checkpoint")
parser.add_argument("--relation", type=str, required=True,
                        help="Relation model checkpoint")
parser.add_argument("--data", type=str, required=True,
                        help="Test data")
parser.add_argument("--n_gpu", type=str, required=True,
                        help="GPU used")
parser.add_argument("--batch_size",type=int,help="Batch size",default=8)
parser.add_argument("--output_dir",type=str,help="Output directory",required=True)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print("Preparing dataset...")

dataset = read_files([args.data],"aste")
dataset["target"] = dataset.target.astype(str)

print("Sample text :",dataset.text.iloc[0])

print("Preparing model and tokenizers...")

concept_tokenizer = AutoTokenizer.from_pretrained(args.concept, local_files_only=True, use_fast=False)
sm_tokenizer = AutoTokenizer.from_pretrained(args.sm, local_files_only=True, use_fast=False)
relation_tokenizer = AutoTokenizer.from_pretrained(args.relation, problem_type=None, local_files_only=True, use_fast=False)

concept_model = AutoModelForTokenClassification.from_pretrained(args.concept, local_files_only=True)
sm_model = AutoModelForTokenClassification.from_pretrained(args.sm, local_files_only=True)
relation_model = AutoModelForSequenceClassification.from_pretrained(args.relation, local_files_only=True)
device = torch.device(f'cuda:{args.n_gpu}')
concept_model.to(device)
sm_model.to(device)
relation_model.to(device)

global_start_time = time.time()

print("Tokenizing text...")

tokenized_text_concept = concept_tokenizer.batch_encode_plus(dataset["text"], max_length=concept_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
tokenized_text_sm = sm_tokenizer.batch_encode_plus(dataset["text"], max_length=sm_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)

concept_data_loader = torch.utils.data.DataLoader(tokenized_text_concept["input_ids"],batch_size=args.batch_size,shuffle=False)
sm_data_loader = torch.utils.data.DataLoader(tokenized_text_sm["input_ids"],batch_size=args.batch_size,shuffle=False)

print("Start predicting for concepts and sentiment markers...")

concept_logits = []
sm_logits = []

concept_model_start_time = time.time()
for batch in tqdm(concept_data_loader):
    concept_logits.extend(concept_model(input_ids=batch).logits.to('cpu').tolist())
concept_model_end_time = time.time()

sm_model_start_time = time.time()
for batch in tqdm(sm_data_loader):
    sm_logits.extend(sm_model(input_ids=batch).logits.to('cpu').tolist())
sm_model_end_time = time.time()

concept_predicted_token_class_ids = np.argmax(concept_logits,axis=-1)
sm_predicted_token_class_ids = np.argmax(sm_logits,axis=-1)

concept_predicted_tokens_classes = [[concept_model.config.id2label[t.item()] for t in el] for el in concept_predicted_token_class_ids]
sm_predicted_tokens_classes = [[sm_model.config.id2label[t.item()] for t in el] for el in sm_predicted_token_class_ids]

concept_token_spans = [catch_token_span(tokens,polarized=False) for tokens in concept_predicted_tokens_classes]
sm_token_spans = [catch_token_span(tokens) for tokens in sm_predicted_tokens_classes]

concept_input_tokens = [postprocess_bio(concept_tokenizer.convert_ids_to_tokens(ids,skip_special_tokens=False)) for ids in tokenized_text_concept["input_ids"]]
sm_input_tokens = [postprocess_bio(sm_tokenizer.convert_ids_to_tokens(ids,skip_special_tokens=False)) for ids in tokenized_text_sm["input_ids"]]

detokenized_concepts = [[detokenize(concept_tokenizer,concept_input_tokens[i],spans) for spans in concept_token_spans[i]] for i in range(len(concept_input_tokens))]
detokenized_sm_pos = [[detokenize(sm_tokenizer,sm_input_tokens[i],spans) for spans in sm_token_spans[i]["POS"]] for i in range(len(sm_input_tokens))]
detokenized_sm_neg = [[detokenize(sm_tokenizer,sm_input_tokens[i],spans) for spans in sm_token_spans[i]["NEG"]] for i in range(len(sm_input_tokens))]

relation_dataset = {
    "text" : [],
    "combination" : [],
    "triplet" : []
}

texts = dataset["text"]

for i in range((len(texts))):
    for j in range(len(detokenized_concepts[i])):
        for k in range(len(detokenized_sm_pos[i])):
            concept = detokenized_concepts[i][j]
            sentiment_marker = detokenized_sm_pos[i][k]
            
            # concept = fixing(texts[i],concept)
            # sentiment_marker = fixing(texts[i],sentiment_marker)

            if concept not in texts[i]:
                from_tokenizer_input = concept_tokenizer.convert_tokens_to_string(concept_input_tokens[i])
                concept = fixing(from_tokenizer_input,detokenized_concepts[i][j],is_subword=True)
            concept = concept if concept != None else detokenized_concepts[i][j]
            concept = fixing(texts[i],concept,is_subword=False)
            if sentiment_marker not in texts[i]:
                from_tokenizer_input = sm_tokenizer.convert_tokens_to_string(sm_input_tokens[i])
                sentiment_marker = fixing(from_tokenizer_input,sentiment_marker,is_subword=True)
            sentiment_marker = sentiment_marker if sentiment_marker != None else detokenized_sm_pos[i][k]
            sentiment_marker = fixing(texts[i],sentiment_marker,is_subword=False)
            combination = f"{concept.lower()}-{sentiment_marker.lower()}"

            if combination not in relation_dataset["combination"]:
                relation_dataset["text"].append(texts[i])
                relation_dataset["combination"].append(combination)
                # relation_dataset["triplet"].append((concept,sentiment_marker,"positive"))
                relation_dataset["triplet"].append({
                    "aspect" : concept,
                    "opinion" : sentiment_marker,
                    "sentiment" : "positive"
                    })
    
    # for j in range(len(detokenized_concepts_neg[i])):
        for k in range(len(detokenized_sm_neg[i])):
            concept = detokenized_concepts[i][j]
            sentiment_marker = detokenized_sm_neg[i][k]

            if concept not in texts[i]:
                from_tokenizer_input = concept_tokenizer.convert_tokens_to_string(concept_input_tokens[i])
                concept = fixing(from_tokenizer_input,detokenized_concepts[i][j],is_subword=True)
            concept = concept if concept != None else detokenized_concepts[i][j]
            concept = fixing(texts[i],concept,is_subword=False)
            if sentiment_marker not in texts[i]:
                from_tokenizer_input = sm_tokenizer.convert_tokens_to_string(sm_input_tokens[i])
                sentiment_marker = fixing(from_tokenizer_input,sentiment_marker,is_subword=True)
            sentiment_marker = sentiment_marker if sentiment_marker != None else detokenized_sm_neg[i][k]
            sentiment_marker = fixing(texts[i],sentiment_marker,is_subword=False)
            combination = f"{concept.lower()}-{sentiment_marker.lower()}"
            
            if combination not in relation_dataset["combination"]:
                relation_dataset["text"].append(texts[i])
                relation_dataset["combination"].append(combination)
                # relation_dataset["triplet"].append((concept,sentiment_marker,"negative"))
                relation_dataset["triplet"].append({
                    "aspect" : concept,
                    "opinion" : sentiment_marker,
                    "sentiment" : "negative"
                    })

relation_dataset = pd.DataFrame(relation_dataset)

tokenized_text_relation = relation_tokenizer.batch_encode_plus(relation_dataset["text"].tolist(),relation_dataset["combination"].tolist(), max_length=relation_model.config.max_length, padding=True, truncation=True, return_tensors="pt").to(device)

relation_data_loader = torch.utils.data.DataLoader(tokenized_text_relation["input_ids"],batch_size=args.batch_size,shuffle=False)
relation_predictions = []

print("Predicting relations...")

relation_model_start_time = time.time()
for batch in tqdm(relation_data_loader):
    relation_predictions.extend(relation_model(input_ids=batch).logits.to('cpu').tolist())
relation_model_end_time = time.time()

global_end_time = time.time()

relation_label_ids = np.argmax(relation_predictions,axis=-1)
relation_labels = [id.item() for id in relation_label_ids]

relation_dataset["is_connected"] = relation_labels

unique_text = relation_dataset.text.unique()

final_result = {
    "text" : dataset["text"],
    "target" : dataset["target"], # .str.lower(),
    "preds" : []
}

for i in range(len(final_result["text"])):
    final_result["target"][i] = eval(final_result["target"][i])
    text = final_result["text"][i]
    triplets = relation_dataset.loc[(relation_dataset["text"] == text) & (relation_dataset["is_connected"] == 1),"triplet"].tolist()
    final_result["preds"].append(triplets)
print("Evaluation...")
evaluation = evaluate(final_result["preds"],final_result["target"])
print(evaluation)
inference_time = {
    "global" : global_end_time - global_start_time,
    "concept" : concept_model_end_time - concept_model_start_time,
    "sentiment_marker" : sm_model_end_time - sm_model_start_time,
    "relation" : relation_model_end_time - relation_model_start_time,
    "n_data" : len(dataset),
    "n_data_relation" : len(relation_dataset)
}
print("Inference time data :",inference_time)
model_info = {
    "concept" : args.concept,
    "sentiment_marker" : args.sm,
    "relation" : args.relation
}
json.dump(inference_time,open(os.path.join(args.output_dir,"inference_time.json"),"w"))
json.dump(evaluation,open(os.path.join(args.output_dir,"evaluation.json"),"w"))
json.dump(model_info,open(os.path.join(args.output_dir,"model_info.json"),"w"))
print("Save result...")
final_result = pd.DataFrame(final_result)
final_result.to_csv(os.path.join(args.output_dir,"prediction.csv"),index=None)
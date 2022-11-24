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

import datasets

import sys
sys.path.append("../gabsa")

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
    

def tokenize_and_align_labels(examples,tokenizer,encoding_args={}):
    encoding_args["is_split_into_words"] = True
    tokenized_inputs = tokenizer(examples["tokens"], **encoding_args)# truncation=True, is_split_into_words=True, max_length=256)

    labels = []
    for i, label in enumerate(examples["bio_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                try:
                    label_ids.append(label[word_idx])
                except Exception as e:
                    print(label)
                    raise e
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

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

concept_tokenizer = AutoTokenizer.from_pretrained(args.concept, local_files_only=True)
sm_tokenizer = AutoTokenizer.from_pretrained(args.sm, local_files_only=True)
relation_tokenizer = AutoTokenizer.from_pretrained(args.relation, problem_type=None, local_files_only=True)

concept_model = AutoModelForTokenClassification.from_pretrained(args.concept, local_files_only=True)
sm_model = AutoModelForTokenClassification.from_pretrained(args.sm, local_files_only=True)
relation_model = AutoModelForSequenceClassification.from_pretrained(args.relation, local_files_only=True)
device = torch.device(f'cuda:{args.n_gpu}')
concept_model.to(device)
sm_model.to(device)
relation_model.to(device)


tokens_and_bio_tags_concept = {"tokens" : [], "bio_tags" : []}
tokens_and_bio_tags_sm = {"tokens" : [], "bio_tags" : []}

# get the tokens and bio tags
for i, row in dataset.iterrows():
    splitted_text = row["text"].split()
    num_target = row["num_target"]

    bio_tags_concept = ['O' for _ in splitted_text]
    bio_tags_sm = bio_tags_concept.copy()

    bio_tags_concept = [concept_model.config.label2id[el] for el in bio_tags_concept]
    bio_tags_sm = [sm_model.config.label2id[el] for el in bio_tags_sm]

    for nt in num_target:
        concept_index = nt[0]
        sm_index = nt[1]
        polarity = nt[2]

        bio_tags_concept[concept_index[0]] = concept_model.config.label2id["B-CONCEPT"]
        bio_tags_sm[sm_index[0]] = sm_model.config.label2id[f"B-{polarity}"]

        for ci in concept_index[1:]:
            bio_tags_concept[ci] = concept_model.config.label2id["I-CONCEPT"]
        for smi in sm_index[1:]:
            bio_tags_sm[smi] = sm_model.config.label2id[f"I-{polarity}"]
    
    tokens_and_bio_tags_concept["tokens"].append(splitted_text)
    tokens_and_bio_tags_sm["tokens"].append(splitted_text)

    tokens_and_bio_tags_concept["bio_tags"].append(bio_tags_concept)
    tokens_and_bio_tags_sm["bio_tags"].append(bio_tags_sm)

tokens_and_bio_tags_concept = datasets.Dataset.from_dict(tokens_and_bio_tags_concept)
tokens_and_bio_tags_sm = datasets.Dataset.from_dict(tokens_and_bio_tags_sm)

global_start_time = time.time()

print("Tokenizing text...")

# tokenized_text_concept = concept_tokenizer.batch_encode_plus(dataset["text"].tolist(), max_length=concept_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
# tokenized_text_sm = sm_tokenizer.batch_encode_plus(dataset["text"].tolist(), max_length=sm_model.config.max_length, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
encoding_args = {
    "max_length":concept_model.config.max_length,
    "padding":True,
    "truncation":True,
    # "add_special_tokens":False,
    "return_tensors":"pt"
}

# def todevice(x,device=device):
#     x['input_ids'] = x['input_ids'].to(device)
#     return x
tokenized_text_concept = tokens_and_bio_tags_concept.map(
    lambda x : tokenize_and_align_labels(x,concept_tokenizer,encoding_args=encoding_args),
    batched=True
)
# tokenized_text_concept = tokenized_text_concept.map(todevice,batched=False)
# tokenized_text_concept['input_ids'] = tokenized_text_concept['input_ids'].to(device)
tokenized_text_sm = tokens_and_bio_tags_sm.map(
    lambda x : tokenize_and_align_labels(x,sm_tokenizer,encoding_args=encoding_args),
    batched=True
)
# tokenized_text_sm = tokenized_text_sm.map(todevice,batched=False)
# tokenized_text_sm['input_ids'] = tokenized_text_sm['input_ids'].to(device)

concept_data_loader = torch.utils.data.DataLoader(tokenized_text_concept["input_ids"],batch_size=args.batch_size,shuffle=False)
sm_data_loader = torch.utils.data.DataLoader(tokenized_text_sm["input_ids"],batch_size=args.batch_size,shuffle=False)

print("Start predicting for concepts and sentiment markers...")

concept_logits = []
sm_logits = []

concept_model_start_time = time.time()
for batch in tqdm(concept_data_loader):
    # input_ids = [el.to(device) for el in batch]
    # print(input_ids)
    # exit()
    # input_ids = [torch.Tensor(el).to(device) for el in input_ids]
    concept_logits.extend(concept_model(input_ids=input_ids).logits.to('cpu').tolist())
concept_model_end_time = time.time()

print(concept_logits[1])
print(len(concept_logits[1]))
print(tokenized_text_concept["input_ids"][1])
print(len(tokenized_text_concept["input_ids"][1]))
print(dataset.loc[1,"text"])
print(len(dataset.loc[1,"text"].split()))
print(concept_model.config.id2label)
print(sm_model.config.id2label)
print(relation_model.config.id2label)
print(tokens_and_bio_tags_concept[1]["bio_tags"])
print(concept_tokenizer)
print(tokenize_and_align_labels(tokens_and_bio_tags_concept[1:2],concept_tokenizer,{"truncation":True,"max_length" : 256}))
exit()

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

texts = dataset["text"].tolist()

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
    "preds" : [],
    "concept" : detokenized_concepts,
    "sm_pos" : detokenized_sm_pos,
    "sm_neg" : detokenized_sm_neg
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

print("EXTRA (predicting relation only)")
# target = dataset["target"]

relation_only = []

for i in range(dataset.shape[0]):
    text = dataset.loc[i,"text"]
    target = dataset.loc[i,"target"]
    if isinstance(target,str):
        target = eval(target)
    all_concept = []
    all_sm = []
    all_pair = []
    for t in target:
        concept = t['aspect'].lower()
        sm = t['opinion'].lower()
        pair = (concept,sm)

        if concept not in all_concept:
            all_concept.append(concept)
        if sm not in all_sm:
            all_sm.append(sm)
        if pair not in all_pair:
            all_pair.append(pair)
    
    for concept in all_concept:
        for sm in all_sm:
            new_entry = {
                "text" : text,
                "pair" : f"{concept}-{sm}"
            }

            if (concept,sm) in all_pair:
                new_entry["label"] = "1"
            else:
                new_entry["label"] = "0"
            
            relation_only.append(new_entry)
    

relation_only = pd.DataFrame(relation_only)

tokenized_text_relation = relation_tokenizer.batch_encode_plus(relation_only["text"].tolist(),relation_only["pair"].tolist(), max_length=relation_model.config.max_length, padding=True, truncation=True, return_tensors="pt").to(device)

relation_data_loader = torch.utils.data.DataLoader(tokenized_text_relation["input_ids"],batch_size=args.batch_size,shuffle=False)
relation_predictions = []

for batch in tqdm(relation_data_loader):
    relation_predictions.extend(relation_model(input_ids=batch).logits.to('cpu').tolist())

relation_label_ids = np.argmax(relation_predictions,axis=-1)
relation_labels = [id.item() for id in relation_label_ids]

relation_only["preds"] = relation_labels

relation_only.to_csv(os.path.join(args.output_dir,"relation_prediction.csv"),index=None)

json.dump(inference_time,open(os.path.join(args.output_dir,"inference_time.json"),"w"))
json.dump(evaluation,open(os.path.join(args.output_dir,"evaluation.json"),"w"))
json.dump(model_info,open(os.path.join(args.output_dir,"model_info.json"),"w"))
print("Save result...")
final_result = pd.DataFrame(final_result)
final_result.to_csv(os.path.join(args.output_dir,"prediction.csv"),index=None)
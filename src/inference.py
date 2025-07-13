import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
import preprocess
import postprocess
from evaluation import summary_score
import json

parser = ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--se_order", type=str, default="aos")
parser.add_argument("--prompt", type=str, default="lego_absa")
parser.add_argument("--answer", type=str, default="lego_absa")
parser.add_argument("--out_path", type=str, default="./out")

args = parser.parse_args()

gpu = args.gpu
model_path = args.model_path
max_length = args.max_length
data_path = args.data_path
se_order = args.se_order
prompt = args.prompt
answer = args.answer
out_path = args.out_path

os.makedirs(out_path, exists_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device("cuda:0")

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(task="text2text-generation",model=model, tokenizer=tokenizer, device=device)

data_reader = preprocess.DataReader()
data_augmentator = preprocess.DataAugmentator()

model.config.max_length = max_length

data = data_reader.do(data_path)
augmented_data = data_augmentator.do(data, 
                                     "aos", 
                                     [{"se_order" : se_order, 
                                       "prompt" : prompt, 
                                       "answer" : answer}], 
                                     1, 
                                     shuffle=False)

answer_catcher = postprocess.AnswerCatcher()
cleaner = postprocess.Cleaner()

catch_answer_fn = getattr(answer_catcher, answer)

inputs = [el["input"] for el in augmented_data]
texts = [el for el in inputs]
targets = [catch_answer_fn(el["output"], el["se_order"], t) for el, t in zip(augmented_data, texts)]

preds = pipe(inputs, return_tensors=True)

raw_preds = tokenizer.batch_decode([el["generated_token_ids"] for el in preds])

preds = cleaner.many(raw_preds, remove=[tokenizer.pad_token, tokenizer.eos_token])

preds = [catch_answer_fn(p, el["se_order"], t) for p, el, t in zip(preds, augmented_data, texts)]

score = summary_score(preds, targets)

with open(os.path.join(out_path, "raw_preds.json"), 'w') as fp:
    json.dump(raw_preds, fp)

with open(os.path.join(out_path, "preds.json"), 'w') as fp:
    json.dump(preds, fp)

with open(os.path.join(out_path, "score.json"), 'w') as fp:
    json.dump(score, fp)

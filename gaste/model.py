# T5 family
from transformers import T5ForConditionalGeneration, T5Tokenizer, ByT5Tokenizer, MT5ForConditionalGeneration

# XGLM
from transformers import XGLMTokenizer, XGLMForCausalLM

# BERT (IndoBERT) => existence of triplet classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_gaste_tokenizer_and_model(model_type,model_name_or_path):
    tokenizer = None
    model = None
    if model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    elif model_type == "byt5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = ByT5Tokenizer.from_pretrained(model_name_or_path)
    elif model_type == "mt5":
        model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    elif model_type == "xglm":
        model = XGLMForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = XGLMTokenizer.from_pretrained(model_name_or_path)
    else:
        raise NotImplementedError
    return {"model" : model, "tokenizer" : tokenizer, "type" : model_type, "model_name_or_path" : model_name_or_path}

def get_triplet_existence_model_and_tokenizer(model_type,model_name_or_path):
    model = None
    tokenizer = None
    if model_type == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=False)
    else:
        raise NotImplementedError
    return {"model" : model, "tokenizer" : tokenizer, "type" : model_type, "model_name_or_path" : model_name_or_path}
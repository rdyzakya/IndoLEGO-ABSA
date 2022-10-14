# T5 family
from transformers import T5ForConditionalGeneration, T5Tokenizer, ByT5Tokenizer, MT5ForConditionalGeneration

# XGLM
from transformers import XGLMTokenizer, XGLMForCausalLM

# BART
from transformers import BartForConditionalGeneration, BartTokenizer

def get_gabsa_tokenizer_and_model(model_type,model_name_or_path,model_args,tokenizer_args):
    tokenizer = None
    model = None
    if model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path,**model_args)
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
    elif model_type == "byt5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path,**model_args)
        tokenizer = ByT5Tokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
    elif model_type == "mt5":
        model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path,**model_args)
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
    elif model_type == "xglm":
        model = XGLMForCausalLM.from_pretrained(model_name_or_path,**model_args)
        tokenizer = XGLMTokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
    elif model_type == "bart":
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path,**model_args)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
    else:
        raise NotImplementedError
    return {"model" : model, "tokenizer" : tokenizer, "type" : model_type, "model_name_or_path" : model_name_or_path}
from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import re
from typing import List, Dict


class ABSAGenerativeModelWrapper:
    """
    Wrapper class containing HuggingFace generative model and tokenizer.
    """
    def __init__(self,model_name_or_path:str,model_args:Dict={},tokenizer_args:Dict={}):
        """
        ### DESC
            Constructor for absa generative model (containing huggingface model and tokenizer). Will detect if the model exist as a Seq2SeqLM Model, if not then the model is a CausalLM Model.
        ### PARAMS
        * model_name_or_path: HuggingFace model name or path.
        * model_args: HuggingFace model keyword arguments.
        * tokenizer_args: HuggingFace tokenizer keyword arguments.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,**tokenizer_args)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,**model_args)
            self.prompt_side = "left"
            self.model_type = "seq2seq"
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,**model_args)
            self.prompt_side = "left" # "right"
            self.model_type = "causal_lm"
        if self.model_type == "causal_lm":
            resize = False
            if self.tokenizer.pad_token == None:
                # self.tokenizer.add_special_tokens({'pad_token': re.sub(r"[a-zA-Z]+","pad",list(self.tokenizer.special_tokens_map.values())[0])})
                pad_token = "<|pad|>"# self.tokenizer.eos_token or self.tokenizer.unk_token
                self.tokenizer.add_tokens([pad_token])
                self.tokenizer.add_special_tokens({"pad_token": pad_token})
                resize = True
            if self.tokenizer.eos_token == None:
                eos_token = "<|endoftext|>"
                self.tokenizer.add_tokens([eos_token])
                self.tokenizer.add_special_tokens({"eos_token": eos_token})
                resize = True
            if self.tokenizer.sep_token == None:
                sep_token = "<|sep|>"
                self.tokenizer.add_tokens([sep_token])
                self.tokenizer.add_special_tokens({"sep_token": sep_token})
                resize = True
            if resize:
                self.model.resize_token_embeddings(len(self.tokenizer))

    def from_pretrained(model_name_or_path:str,**kwargs) -> ABSAGenerativeModelWrapper:
        """
        ### DESC
            Class method returning absa generative model (containing huggingface model and tokenizer). Will detect if the model exist as a Seq2SeqLM Model, if not then the model is a CausalLM Model.
        ### PARAMS
        * model_name_or_path: HuggingFace model name or path.
        * kwargs: HuggingFace model or tokenizer keyword arguments.
        ### RETURN
        * ABSAGenerativeModel instance.
        """
        return ABSAGenerativeModelWrapper(model_name_or_path,**kwargs)
    
    def to(self,device:torch.device=torch.device("cpu")) -> ABSAGenerativeModelWrapper:
        """
        ### DESC
            Method to place the model to the designated computing device (cuda or cpu).
        ### PARAMS
        * device: torch.device instance.
        ### RETURN
        * The model and tokenizer wrapper it self.
        """
        self.model.to(device)
        return self
    
    def add_vocab(self,new_vocab:List[str]):
        """
        ### DESC
            Method to add new vocabularies to tokenizer.
        ### PARAMS
        * new_vocab: List of new vocabularies.
        """
        try:
            vocab = self.tokenizer.get_vocab()
        except NotImplementedError:
            vocab = self.tokenizer.vocab
        for term in new_vocab:
            tokenized_term = self.tokenizer.tokenize(term)
            for token in tokenized_term:
                if token not in vocab:
                    self.tokenizer.add_tokens(token)

if __name__ == "__main__":
    wrapper = ABSAGenerativeModelWrapper("t5-small")
    print(wrapper.model)
    print(wrapper.tokenizer)
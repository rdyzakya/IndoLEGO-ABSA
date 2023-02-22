from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch

class ABSAGenerativeModelWrapper:
    """
    Wrapper class containing HuggingFace generative model and tokenizer.
    """
    def __init__(self,model_name_or_path:str,**kwargs):
        """
        ### DESC
            Constructor for absa generative model (containing huggingface model and tokenizer). Will detect if the model exist as a Seq2SeqLM Model, if not then the model is a CausalLM Model.
        ### PARAMS
        * model_name_or_path: HuggingFace model name or path.
        * kwargs: HuggingFace model or tokenizer keyword arguments.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,**kwargs)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,**kwargs)
            self.prompt_side = "left"
            self.model_type = "seq2seq"
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,**kwargs)
            self.prompt_side = "right"
            self.model_type = "causal_lm"

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
    
    def to(self,device:torch.device=torch.device("cpu")):
        """
        ### DESC
            Method to place the model to the designated computing device (cuda or cpu).
        ### PARAMS
        * device: torch.device instance.
        """
        self.model.to(device)
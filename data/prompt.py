from typing import List, Dict
from pattern import Pattern
import json

class Prompter:
    """
    Prompter for adding prompts to texts.
    """
    def __init__(self,pattern:Pattern=Pattern()):
        """
        ### DESC
            Constructor for Prompter object.
        ### PARAMS
        * pattern: Pattern object (may be used in prompts).
        """
        self.pattern = pattern

    def build_prompt(self,template:str,task:str="acos",dict_target:List[Dict]=[]) -> str:
        # Extract in the form of (A, O, C, S) with the category {CAT1, CAT2} : TEXT
        # Extract in the format PATTERN with the category CATEGORY : TEXT
        # Impute (pizza, O, C, positive) ; (drink, O, C, positive) with the category {CAT1, CAT2} : TEXT
        pass


    # def compile(self,prompt_side,option):
    #     if option not in self.available_option:
    #         raise ValueError(f"Option should only be from {self.available_option}")
    #     if prompt_side != "left" and prompt_side != "right":
    #         raise ValueError(f"Prompt side should only be 'left' or 'right'")
    #     self.prompt_side = prompt_side
    #     self.option = option
    
    # def add_prompt(self,task,texts,option):
    #     if option not in self.available_option:
    #         raise ValueError(f"Option should only be from {self.available_option}")
    #     if option == "no_prompt":
    #         return texts, ["" for _ in texts]
    #     prompt_side = self.prompt_side
    #     chosen_prompt = None
    #     prompts = []
    #     if isinstance(option,int):
    #         chosen_prompt = self.prompts[task][option]
    #         prompts = [chosen_prompt for _ in texts]
    #     result = []
    #     for t in texts:
    #         if option == "random":
    #             chosen_prompt = random.choice(self.prompts[task])
    #             prompts.append(chosen_prompt)
    #         t = chosen_prompt + ' ' + t if prompt_side == "left" else t + ' ' + chosen_prompt
    #         result.append(t)
    #     return result, prompts
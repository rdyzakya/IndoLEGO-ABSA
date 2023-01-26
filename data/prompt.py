from pattern import Pattern
import json

class Prompter:
    """
    Prompter for adding prompts to texts.
    """
    def __init__(self,prompt_path:str,prompt_side:str="left",pattern:Pattern=Pattern()):
        """
        ### DESC
            Constructor for Prompter object.
        ### PARAMS
        * prompt_path: Path to the json file consist of prompts. Available prompt keys are "extraction", "imputation", "no_prompt"
        * prompt_side: Side for adding the prompt. Available sides are "left" and "right".
        * pattern: Pattern object (may be used in prompts).
        """
        self.prompts = json.load(open(prompt_path,'r',encoding="utf-8"))
        assert set(self.prompts.keys()).issubset({"extraction", "imputation", "no_prompt"})

        assert prompt_side == "left" or prompt_side == "right"
        self.prompt_side = prompt_side
        
        self.pattern = pattern

    def build_prompt(self):
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
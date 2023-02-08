from typing import List, Dict
from pattern import Pattern
from constant import FORMAT_PATTERN_MASK, CATEGORY_MASK, IMPUTATION_FIELD_MASK

default_template = {
    "extraction" : f"Extract with the format {FORMAT_PATTERN_MASK} with the categories {CATEGORY_MASK} for the following text",
    "imputation" : f"Impute the following {IMPUTATION_FIELD_MASK} for the following text"
}

class Prompter:
    """
    Prompter for adding prompts to texts.
    """
    def __init__(self,prompt_template:Dict=default_template):
        """
        ### DESC
            Constructor for prompter object.
        ### PARAMS
        * prompt_template: Dictionary for prompt templates (consist of 'extraction' or 'imputation' key only).
        """
        paradigms = set(prompt_template.keys())
        assert paradigms.issubset({"extraction", "imputation"})

        self.prompt_template = prompt_template

    def build_prompt(self,task:str="acos",pattern:Pattern=Pattern(),incomplete_result:List[Dict]=[],paradigm:str="extraction") -> str:
        """
        ### DESC
            Method for building prompt.
        ### PARAMS
        * task: Task name. Example: ao, ac, cs, as, aos, acos, etc.
        * pattern: Pattern object.
        * incomplete_result: Tuples that need to be impute (Used if paradigm is 'imputation').
        * paradigm: The paradigm, either extraction or imputation.
        ### RETURN
        * prompt: Resultant prompt.
        """
        # EXAMPLES:
        # Template => Extract in the format PATTERN with the category CATEGORY
        # Resultant prompt 1 => Extract in the form of (A, O, C, S) with the category [CAT1, CAT2] :
        # Resultant prompt 2 => Impute (pizza, O, C, positive) ; (drink, O, C, positive) with the category [CAT1, CAT2] :
        assert paradigm == "extraction" or paradigm == "imputation"

        template = self.prompt_template[paradigm]
        if paradigm == "extraction":
            format_pattern = pattern.pattern[task]
            prompt = template.replace(FORMAT_PATTERN_MASK,format_pattern)
        elif paradigm == "imputation":
            stringified_incomplete_result = [pattern.stringify(d_t,task) for d_t in incomplete_result]
            stringified_incomplete_result = f" {pattern.inter_sep} ".join(stringified_incomplete_result)
            prompt = template.replace(IMPUTATION_FIELD_MASK,stringified_incomplete_result)
        
        categories = pattern.categories
        stringified_categories = str(categories).replace("'",'') # remove the quote
        prompt = prompt.replace(CATEGORY_MASK,stringified_categories) + ": "

        return prompt
    
    def add_prompt(self,prompt:str,text:str,side:str="left") -> str:
        """
        ### DESC
            Method for adding prompt to the designated text.
        ### PARAMS
        * prompt: Added prompt.
        * text: The text that needed to be prompt.
        * side: Where the prompt needs to be placed, either 'left' or 'right'.
        ### RETURN
        * prompted_text: Prompted text.
        """
        assert side == "left" or side == "right"

        prompted_text = prompt + text if side == "left" else text + prompt

        return prompted_text

if __name__ == "__main__":
    pattern = Pattern(categories=["LAPTOP#GENERAL","BATTERY#HEALTH"])
    prompter = Prompter()

    task = "aocs"
    incomplete_result = [{"aspect" : "build quality", "opinion" : "strong"}, {"aspect" : "power", "opinion" : "long enough"}]

    result_1 = prompter.build_prompt(task,pattern,incomplete_result,"extraction")
    result_2 = prompter.build_prompt(task,pattern,incomplete_result,"imputation")

    print(result_1)
    print(result_2)
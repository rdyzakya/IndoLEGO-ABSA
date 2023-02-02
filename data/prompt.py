from typing import List, Dict
from pattern import Pattern

FORMAT_PATTERN_MASK = "PATTERN"
CATEGORY_MASK = "CATEGORY"
IMPUTATION_FIELD_MASK = "IMPUTATION_FIELD"

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

    def build_prompt(self,template:str,task:str="acos",uncomplete_result:List[Dict]=[],paradigm:str="extraction") -> str:
        """
        ### DESC
            Method for building prompt
        ### PARAMS
        * template: Template text for the prompt.
        * task: Task name. Example: ao, ac, cs, as, aos, acos, etc.
        * uncomplete_result: Tuples that need to be impute.
        * paradigm: The paradigm, either extraction or imputation.
        ### RETURN
        * prompt: Resultant prompt.
        """
        # EXAMPLES:
        # Template => Extract in the format PATTERN with the category CATEGORY
        # Resultant prompt 1 => Extract in the form of (A, O, C, S) with the category [CAT1, CAT2] :
        # Resultant prompt 2 => Impute (pizza, O, C, positive) ; (drink, O, C, positive) with the category [CAT1, CAT2] :
        assert paradigm == "extraction" or paradigm == "imputation"

        if paradigm == "extraction":
            format_pattern = self.pattern.pattern[task]
            prompt = template.replace(FORMAT_PATTERN_MASK,format_pattern)
        elif paradigm == "imputation":
            stringified_uncomplete_result = [self.pattern.stringify(d_t,task) for d_t in uncomplete_result]
            stringified_uncomplete_result = f" {self.pattern.inter_sep} ".join(stringified_uncomplete_result)
            prompt = template.replace(IMPUTATION_FIELD_MASK,stringified_uncomplete_result)
        
        categories = self.pattern.categories
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
    prompter = Prompter(pattern=Pattern(categories=["LAPTOP#GENERAL","BATTERY#HEALTH"]))
    template_1 = f"Extract with the format {FORMAT_PATTERN_MASK} for the following text"
    template_2 = f"Extract with the format {FORMAT_PATTERN_MASK} with the categories {CATEGORY_MASK} for the following text"
    template_3 = f"Impute the following {IMPUTATION_FIELD_MASK} for the following text"

    text = "THIS IS A TEXT."
    task = "aocs"
    uncomplete_result = [{"aspect" : "build quality", "opinion" : "strong"}, {"aspect" : "power", "opinion" : "long enough"}]

    result_1 = prompter.add_prompt(
        prompt=prompter.build_prompt(template_1,task,uncomplete_result,"extraction"),
        text=text
    )
    result_2 = prompter.add_prompt(
        prompt=prompter.build_prompt(template_2,task,uncomplete_result,"extraction"),
        text=text
    )
    result_3 = prompter.add_prompt(
        prompt=prompter.build_prompt(template_3,task,uncomplete_result,"imputation"),
        text=text
    )

    print("RESULT 1")
    print(result_1 + "\n")

    print("RESULT 2")
    print(result_2 + "\n")

    print("RESULT 3")
    print(result_3 + "\n")
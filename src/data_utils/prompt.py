from typing import List, Dict
from .pattern import Pattern

class Prompter:
    """
    Prompter for adding prompts to texts.
    """
    def __init__(self, template, place_holder):
        """
        ### DESC
            Constructor for prompter object.
        ### PARAMS
        * template: Prompt template, dictionary containing extraction, imputation, and may be non_absa key.
        * place_holder: Place holder mask (pattern, imputation, category, text).
        """
        paradigms = set(template.keys())
        self.template = template
        self.place_holder = place_holder
    
    def build_prompt(self,pattern:Pattern,task:str="acos",incomplete_result:List[Dict]=[],paradigm:str="extraction") -> str:
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

        template = self.template[paradigm]
        if paradigm == "extraction":
            format_pattern = pattern.template[task]["input"]
            prompt = template.replace(self.place_holder["pattern"],format_pattern)
        elif paradigm == "imputation":
            stringified_incomplete_result = pattern.batch_stringify(incomplete_result,task)
            prompt = template.replace(self.place_holder["imputation"],stringified_incomplete_result)
        
        categories = pattern.categories
        stringified_categories = str(categories).replace("'",'') # remove the quote
        prompt = prompt.replace(self.place_holder["category"],stringified_categories) + ": "

        return prompt
    
    def add_prompt(self,prompt:str,text:str) -> str:
        """
        ### DESC
            Method for adding prompt to the designated text.
        ### PARAMS
        * prompt: Added prompt.
        * text: The text that needed to be prompt.
        ### RETURN
        * prompted_text: Prompted text.
        """
        prompted_text = prompt.replace(self.place_holder["text"], text)

        return prompted_text

if __name__ == "__main__":
    pattern_template = {
        "acos" : {
            "input" : "aspect : <extra_id_0> , category : <extra_id_1> , opinion : <extra_id_2> , sentiment : <extra_id_3>",
            "output" : "<extra_id_0> ASPECT <extra_id_1> CATEGORY <extra_id_2> OPINION <extra_id_3> SENTIMENT"
        },
        "aos" : {
            "input" : "aspect : <extra_id_0> , opinion : <extra_id_2> , sentiment : <extra_id_3>",
            "output" : "<extra_id_0> ASPECT <extra_id_1> OPINION <extra_id_3> SENTIMENT"
        },
        "ao" : {
            "input" : "aspect : <extra_id_0> , opinion : <extra_id_2>",
            "output" : "<extra_id_0> ASPECT <extra_id_1> OPINION"
        }
    }
    pattern_place_holder = {
        "aspect" : "ASPECT",
        "opinion" : "OPINION",
        "category" : "CATEGORY",
        "sentiment" : "SENTIMENT"
    }
    seperator = ';'
    pattern = Pattern(template=pattern_template, place_holder=pattern_place_holder, seperator=seperator)
    
    prompt_template = {
        "extraction" : "Extract with the format PATTERN with the categories CATEGORY for the following text: TEXT",
        "imputation" : "Impute the following IMPUTATION for the following text: TEXT",
        "non_absa" : "NON_ABSA_PROMPT: TEXT"
    }
    
    prompt_place_holder = {
        "pattern" : "PATTERN",
        "category" : "CATEGORY",
        "imputation" : "IMPUTATION",
        "text" : "TEXT",
        "non_absa_prompt" : "NON_ABSA_PROMPT"
    }
    prompter = Prompter(template=prompt_template, place_holder=prompt_place_holder)

    task = "aos"
    incomplete_result = [{"aspect" : "build quality", "opinion" : "strong"}, {"aspect" : "power", "opinion" : "long enough"}]

    result_1 = prompter.build_prompt(pattern,task,incomplete_result,"extraction")
    result_2 = prompter.build_prompt(pattern,task,incomplete_result,"imputation")

    print(result_1)
    print()
    print(result_2)
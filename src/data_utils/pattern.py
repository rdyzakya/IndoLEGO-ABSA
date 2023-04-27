from typing import List, Dict
from .constant import NO_TARGET, SPECIAL_CHAR, SENTTAG2WORD
import re

class Pattern:
    """
    Pattern for the generated answers.
    """
    def __init__(self, template:Dict[str,Dict], place_holder:Dict[str,str], seperator:str, categories:List[str]=[]):
        """
        ### DESC
            Constructor for Pattern objects (pattern for the generated answers).
        ### PARAMS
            * template: Dictionary containing keys (task name) and values (pattern template dictionary with input and output key).
            * place_holder: Place holder for the sentiment elements (key: aspect, opinion, category, sentiment).
            * seperator: Seperator for multiple results.
            * categories: Category list.
        """
        self.template = template
        self.place_holder = place_holder
        self.seperator = seperator
        self.categories = categories
    
    def stringify(self,target:Dict,task:str) -> str:
        """
        ### DESC
            Stringify the target from dictionary form to string form.
        ### PARAMS
        * target: Target in the form of dictionary. Example: {"aspect" : "A1", "opinion" : "O1}
        * task: Task related to the target that needs to be generated.
        ### RETURN
        * output: Stringified target.
        """
        output = self.template[task]["output"]
        for key, value in target.items():
            output = output.replace(self.place_holder[key],value)
        return output
    
    def batch_stringify(self,targets:List[Dict],task:str) -> str:
        """
        ### DESC
            Stringify list of target.
        ### PARAMS
        * targets: List of target.
        * task: Task related to the target that needs to be generated.
        ### RETURN
        * result: Stringified target.
        """
        if len(targets) == 0:
            return NO_TARGET
        result = f" {self.seperator} ".join([self.stringify(target,task) for target in targets])
        return result
    
    def regex(self,task:str) -> str:
        """
        ### DESC
            Method returning the regex form for the related task.
        ### PARAMS
        * task: Task name. Example: aocs, aoc, ac, ao, etc.
        ### RETURN
        * regex_pattern: Regex pattern for the related task.
        """
        regex_pattern = self.template[task]["output"]
        seperator = self.seperator
        for k,v in SPECIAL_CHAR.items():
            regex_pattern = regex_pattern.replace(k,v)
            # intra_sep = intra_sep.replace(k,v)
        for k,v in self.place_holder.items():
            if k == "sentiment":
                regex_pattern = regex_pattern.replace(v,f"(?P<sentiment>{'|'.join(SENTTAG2WORD.values())})")
            else:
                regex_pattern = regex_pattern.replace(v,f"(?P<{k}>[^{seperator}]+)")
        regex_pattern = regex_pattern.replace(' ',r'\s*')
        return regex_pattern
    
    def find_all(self,text:str,task:str) -> List[Dict]:
        """
        ### DESC
            Method to find all stringified tuples in text and transform it back to list of dictionary.
        ### PARAMS
        * text: Text.
        * task: The designated absa task.
        ### RETURN
        """
        regex_pattern = self.regex(task)
        found = [found_iter.groupdict() for found_iter in re.finditer(regex_pattern,text)]
        result = []
        for i in range(len(found)):
            is_found_pattern_token = False
            for k,v in found[i].items():
                # Strip the value
                v = v.strip()
                # Check if the value is between the pattern tokens (usually in prompts)
                if v in self.place_holder.values():
                    is_found_pattern_token = True
                found[i][k] = v
            if found[i] not in result and not is_found_pattern_token:
                result.append(found[i])
        return result
    
    def update_categories(self,categories:List[str]=["CAT0","CAT1"]):
        """
        ### DESC
            Method to update categories attribute.
        ### PARAMS
        * categories: List of category.
        """
        self.categories = categories
    
    def __repr__(self) -> str:
        return str(self.template)
    
    def __str__(self) -> str:
        return str(self.template)

if __name__ == "__main__":
    template = {
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

    place_holder = {
        "aspect" : "ASPECT",
        "opinion" : "OPINION",
        "category" : "CATEGORY",
        "sentiment" : "SENTIMENT"
    }

    seperator = ';'

    p = Pattern(template=template, place_holder=place_holder, seperator=seperator)
    # print(p)
    print(p.regex("acos"))

    # Text input wajib dipotong untuk causal lm
    # text_1 = "Hello semua ayo kita sambut . Extract in the format aspect : <extra_id_0> , opinion : <extra_id_2> : <extra_id_0> Hello <extra_id_1> Semua ; <extra_id_0> Ayo <extra_id_1> sambut"
    text_1 = "<extra_id_0> Hello <extra_id_1> Semua ; <extra_id_0> Ayo <extra_id_1> sambut"
    text_2 = "<extra_id_0> Hallo <extra_id_1> Semua <extra_id_3> positive ; <extra_id_0> Ayo <extra_id_1> sambut <extra_id_3> negative"

    found_1 = p.find_all(text_1,"ao")
    found_2 = p.find_all(text_2,"aos")

    print("Found 1:",found_1)
    print("Found 2:",found_2)
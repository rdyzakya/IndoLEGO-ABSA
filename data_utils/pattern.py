from typing import List, Dict
from constant import SENTTAG2WORD, SPECIAL_CHAR, PATTERN_TOKEN, SENTIMENT_ELEMENT, NO_TARGET
import re

class Pattern:
    """
    Pattern for the generated answers.
    """
    def __init__(self,tasks:List[str]=["ao","ac","as","aos","acs","aocs"],open_bracket:str='(',close_bracket:str=')',intra_sep:str=',',inter_sep:str=';',categories:List[str]=["CAT0","CAT1"]):
        """
        ### DESC
            Constructor for Pattern objects (pattern for the generated answers).
        ### PARAMS
        * task: Tasks used. Example: aocs, aoc, ao, ac, etc.
        * open_bracket: Open bracket symbol.
        * close_bracket: Close bracket symbol.
        * intra_sep: Seperator between sentiment elements in a tuple.
        * inter_sep: Seperator between multiple tuples.
        * categories: List of categories if exist.
        """
        self.tasks = tasks
        self.open_bracket = open_bracket.strip()
        self.close_bracket = close_bracket.strip()
        self.intra_sep = intra_sep.strip()
        self.inter_sep = inter_sep.strip()
        self.categories = categories

        self.mask = {
            self.open_bracket : "/OB/",
            self.close_bracket : "/CB/",
            self.intra_sep : "/AS/",
            self.inter_sep : "/ES/"
        }

        self.compile_tasks(self.tasks)

    def compile_tasks(self,task:List[str]):
        """
        ### DESC
            Method to compile list of tasks handled by the pattern object.
        ### PARAMS
        * task: List of task name.
        """
        self.pattern = {}
        for t in task:
            self.pattern[t] = []
            for se in t:
                # se: a (aspect term), o (opinion term), s (sentiment), c (category)
                self.pattern[t].append(PATTERN_TOKEN[SENTIMENT_ELEMENT[se]])
            self.pattern[t] = f"{self.open_bracket} " + f" {self.intra_sep.strip()} ".join(self.pattern[t]) + f" {self.close_bracket}"
            self.pattern[t] = self.pattern[t].strip()

    def regex(self,task:str) -> str:
        """
        ### DESC
            Method returning the regex form for the related task.
        ### PARAMS
        * task: Task name. Example: aocs, aoc, ac, ao, etc.
        ### RETURN
        * regex_pattern: Regex pattern for the related task.
        """
        regex_pattern = self.pattern[task]
        intra_sep = self.intra_sep
        inter_sep = self.inter_sep
        for k,v in SPECIAL_CHAR.items():
            regex_pattern = regex_pattern.replace(k,v)
            intra_sep = intra_sep.replace(k,v)
        for k,v in PATTERN_TOKEN.items():
            if k == "sentiment":
                regex_pattern = regex_pattern.replace(v,f"(?P<sentiment>{'|'.join(SENTTAG2WORD.values())})")
            elif k == "category":
                regex_pattern = regex_pattern.replace(v,f"(?P<category>{'|'.join(self.categories)})")
            else:
                regex_pattern = regex_pattern.replace(v,f"(?P<{k}>[^{intra_sep}{inter_sep}]+)")
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
                if v in PATTERN_TOKEN.values():
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

    def stringify(self,target:Dict,task:str) -> str:
        """
        ### DESC
            Stringify the target from dictionary form to string form.
        ### PARAMS
        * target: Target in the form of dictionary. Example: {"aspect" : "A1", "opinion" : "O1}
        * task: Task related to the target that needs to be generated.
        ### RETURN
        * result: Stringified target.
        """
        result = self.pattern[task]
        for k in target.keys():
            result = result.replace(PATTERN_TOKEN[k],target[k])
        return result
    
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
        result = f" {self.inter_sep} ".join([self.stringify(target,task) for target in targets])
        return result
    
    def masking(self,text:str) -> str:
        """
        ### DESC
            Method for masking the special characters in the text (open bracket symbol, close bracket symbol, etc.).
        ### PARAMS
        * text: The text that needs to be masked.
        ### RETURN
        * text: Masked text.
        """
        for character, mask in self.mask.items():
            text = text.replace(character,mask)
        return text
    
    def unmasking(self,text:str) -> str:
        """
        ### DESC
            Method for unmasking a text.
        ### PARAMS
        * text: Text that needs to be unmask.
        ### RETURN
        * text: Unmasked text.
        """
        for character, mask in self.mask.items():
            text = text.replace(mask,character)
        return text

    def __repr__(self) -> str:
        return str(self.pattern)
    
    def __str__(self) -> str:
        return str(self.pattern)

if __name__ == "__main__":
    p = Pattern(["aocs", "aoc", "caos", "ao", "a", "aos"])
    print(p)
    print(p.regex("aocs"))
    text_1 = "Hello semua ayo kita sambut . Extract in the format ( <A> , <O> ) : (Hello, semua) ; (ayo, sambut)"
    text_2 = "(Hello, semua, positive) ; (ayo, sambut, negative)"

    found_1 = p.find_all(text_1,"ao")
    found_2 = p.find_all(text_2,"aos")

    print("Found 1:",found_1)
    print("Found 2:",found_2)
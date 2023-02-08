from typing import List, Dict
from constant import SENTTAG2WORD, SPECIAL_CHAR, PATTERN_TOKEN, SENTIMENT_ELEMENT

class Pattern:
    """
    Pattern for the generated answers.
    """
    def __init__(self,task:List[str]=["ao","ac","as","aos","acs","aocs"],open_bracket:str='(',close_bracket:str=')',intra_sep:str=',',inter_sep:str=';',categories:List[str]=["CAT0","CAT1"]):
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
        self.task = task
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

        self.pattern = {}
        for t in self.task:
            self.pattern[t] = []
            for se in t:
                # se: a (aspect term), o (opinion term), s (sentiment), c (category)
                self.pattern[t].append(PATTERN_TOKEN[SENTIMENT_ELEMENT[se]])
            self.pattern[t] = f"{open_bracket} " + f" {intra_sep.strip()} ".join(self.pattern[t]) + f" {close_bracket}"
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
    
    def update_categories(self,categories:List[str]=["CAT0","CAT1"]):
        """
        ### DESC
            Method to update categories attribute.
        ### PARAMS
        * categories: List of category.
        """
        self.categories = categories

    def stringify(self,dict_target:Dict,task:str) -> str:
        """
        ### DESC
            Stringify the target from dictionary form to string form.
        ### PARAMS
        * dict_target: Target in the form of dictionary. Example: {"aspect" : "A1", "opinion" : "O1}
        * task: Task related to the target that needs to be generated.
        ### RETURN
        * result: Stringified target.
        """
        result = self.pattern[task]
        for k in dict_target.keys():
            try:
                result = result.replace(PATTERN_TOKEN[k],dict_target[k])
            except Exception as e:
                print(dict_target)
                raise e
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
    p = Pattern(["aocs", "aoc", "caos", "ao", "a"])
    print(p)
    print(p.regex("aocs"))
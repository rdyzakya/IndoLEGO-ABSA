import sys
sys.path.append("..")
import constant
from typing import Dict, List

class DataReader:
    """
    Responsible to read txt files containing ABSA dataset.
    """
    def do(self, path:str) -> List[Dict]:
        """
        ### DESC
            Method to read dataset. Each line is in the format of TEXT####TARGETS .
        ### PARAMS
        * path: Data path.
        ### RETURN
        * Dataset containing "text" and "num_targets" key.
        """
        assert path.endswith(".txt")
        with open(path, 'r') as reader:
            data = reader.read().strip().splitlines()
        for i,line in enumerate(data):
            try:
                text, num_targets = line.split(constant.SEP)
            except Exception as e:
                raise ValueError(f"Each line should be in the format 'TEXT{constant.SEP}TARGET'. Yours: {line}")
            num_targets = eval(num_targets)
            data[i] = {"text" : text, "num_targets" : num_targets}
        return data
    
    def __call__(self, path:str) -> List[Dict]:
        """
        ### DESC
            Read dataset. Each line is in the format of TEXT####TARGETS .
        ### PARAMS
        * path: Data path.
        ### RETURN
        * Dataset containing "text" and "num_targets" key.
        """
        return self.do(path=path)
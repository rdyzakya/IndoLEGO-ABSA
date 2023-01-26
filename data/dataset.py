from datasets import Dataset, DatasetDict
import pandas as pd

from typing import Dict, List

# THINGS TO DO
# a. Consider tasks
# b. Consider prompter
# c. Consider random shuffle
# d. Consider extraction & imputation
# e. Multiply the data or not

SEP = "####"
sample = "It rarely works and when it does it 's incredibly slow .####[([2], [1], 'NEG')]"

class ABSADataset(Dataset):
    """
    ABSA Dataset.
    """
    def __init__(self,data_path:str,target_format:str,tasks:Dict[str,List],multiply:bool=True,shuffle:bool=True,random_state:int=None):
        """
        ### DESC
            Constructor method for ABSA dataset.
        ### PARAMS
        * data_path: Path to ABSA txt dataset.
        * target_format: Format of the targets in the dataset file. Example: ao, aosc, aos, etc.
        * tasks: Dictionary for available tasks. Available keys are 'extraction' and 'imputation'. Task names indicates the sentiment elements to be extracted, available sentiment elements are Aspect Term (a or A), Opinion Term (o or O), Sentiment (s or S), and Category (c or C). Example of task name: aoc, aos, aosc, ao, etc.
        * multiply: Multiply the dataset (True) or randomly assign random task to the data (uniform distribution).
        * shuffle: Shuffle the dataset.
        * random_state: Seed for randomize the shuffle (only works when shuffle equals True).
        """
        super().__init__()
        # Assert data type
        assert isinstance(data_path,str) and isinstance(target_format,str) \
            and isinstance(tasks,dict) and isinstance(multiply,bool) \
            and isinstance(multiply,bool) and isinstance(shuffle,bool) \
            and isinstance(random_state,int)

        # Assert key of tasks
        tasks_key = set(tasks.keys())
        assert tasks_key.issubset({"extraction", "imputation"})

        # Read the data
        with open(data_path,'r') as reader:
            data = reader.read().strip().splitlines()
        for i,line in enumerate(data):
            try:
                text, num_targets = line.split(SEP)
                num_targets = eval(num_targets)
            except Exception as e:
                raise ValueError(f"Each line should be in the format 'TEXT{SEP}TARGET. Example: {sample}'")
            data[i] = {"text" : text, "num_targets" : num_targets}


a = ABSADataset('test.txt',{'imputation' : ['ao']})
ABSADataset
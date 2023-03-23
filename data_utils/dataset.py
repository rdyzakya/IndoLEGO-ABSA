from datasets import Dataset
from .pattern import Pattern
from .prompt import Prompter
import pandas as pd
import random

from typing import Dict, List

from constant import SEP, SENTTAG2WORD, SENTIMENT_ELEMENT, IMPLICIT_ASPECT

sample = "It rarely works and when it does it 's incredibly slow .####[([2], [1], 'NEG')]"

def remove_duplicate_targets(targets:List[Dict]) -> List[Dict]:
    """
    ### DESC
        Method for removing duplicates in targets.
    ### PARAMS
    * targets: List of target dictionary.
    ### RETURN
    * result_targets: Resulting targets.
    """
    result_targets = []
    for target in targets:
        if target not in result_targets:
            result_targets.append(target)
    return result_targets

def handle_mix_sentiment(targets:List[Dict]) -> List[Dict]: ### MAY CONTAIN BUG, AFFECTS ORDER
    """
    ### DESC
        Method to preprocess targets (especially to reduce target containing sentiments).
    ### 
    * targets: List of targets.
    ### RETURN
    * result_targets: Resulting targets.
    """
    targets_copy = [target.copy() for target in targets]
    results_targets = []
    non_sentiment_target_stack = []
    sentiment_target_stack = []
    for target in targets_copy:
        # Filter targets without sentiment key
        if "sentiment" not in target.keys():
            results_targets.append(target)
        else:
            sentiment_target_stack.append(target["sentiment"])
            del target["sentiment"]
            non_sentiment_target_stack.append(target)
    for target in results_targets:
        targets_copy.remove(target)
    while len(non_sentiment_target_stack) > 0:
        non_sentiment_target = non_sentiment_target_stack.pop(0)
        sentiment_target = sentiment_target_stack.pop(0)
        if non_sentiment_target not in non_sentiment_target_stack:
            target = non_sentiment_target.copy()
            target["sentiment"] = sentiment_target
            results_targets.append(target)
        else:
            sentiments = [sentiment_target]
            for i in range(len(non_sentiment_target_stack)):
                if non_sentiment_target == non_sentiment_target_stack[i]:
                    sentiments.append(sentiment_target_stack[i])
            sentiments = list(set(sentiments))
            if "neutral" in sentiments:
                sentiments.remove("neutral")
            if ("positive" in sentiments and "negative" in sentiments) or "mixed" in sentiments:
                target = non_sentiment_target.copy()
                target["sentiment"] = "mixed"
                results_targets.append(target)
            else:
                target = non_sentiment_target.copy()
                target["sentiment"] = sentiments[0] if len(sentiments) > 0 else "neutral"
                results_targets.append(target)
            while non_sentiment_target in non_sentiment_target_stack:
                non_sentiment_target_index = non_sentiment_target_stack.index(non_sentiment_target)
                non_sentiment_target_stack.pop(non_sentiment_target_index)
                sentiment_target_stack.pop(non_sentiment_target_index)
    return results_targets

class CustomDataset:
    """
    Custom dataset.
    """
    def __init__(self):
        self.data = []

    def __len__(self) -> int:
        try:
            return len(self.data)
        except:
            return -1

class ABSADataset(CustomDataset):
    """
    ABSA Dataset.
    """
    def __init__(self,data_path:str,target_format:str="acos",prompter:Prompter=Prompter(),prompt_side:str="left",pattern:Pattern=Pattern()):
        """
        ### DESC
            Constructor method for ABSA dataset.
        ### PARAMS
        * data_path: Path to ABSA txt dataset.
        * target_format: Format of the targets in the dataset file. Example: ao, aosc, aos, etc.
        * prompter: Prompter object to add prompt.
        * prompt_side: Prompt side, either 'left' or 'right'.
        * pattern: Pattern object.
        """
        super().__init__()
        # Assert data type
        assert isinstance(data_path,str)
        assert isinstance(target_format,str)
        assert isinstance(prompter,Prompter)
        assert isinstance(prompt_side,str)
        assert isinstance(pattern,Pattern)

        # Assert prompt side
        assert prompt_side in ["left","right"]

        # Assign attributes
        self.data_path = data_path
        self.target_format = target_format
        self.prompter = prompter
        self.prompt_side = prompt_side
        self.pattern = pattern

        # Read the data
        self.data = self.read_data(data_path)
    
    def build_train_val_data(self,tasks:Dict={"extraction" : ["ao","ac","as"],"imputation" : {"acos" : ["ao","ac","aos"]}},multiply:bool=True,shuffle:bool=True,random_state:int=None) -> Dataset:
        """
        ### DESC
            Method to build train or validation dataset (HF Dataset).
        ### PARAMS
        * tasks: Task dictionary.
        * multiply: If true then multiply the dataset according to the number of task, else using round robbin manner to shuffle the task for the dataset.
        * shuffle: If true then shuffle the dataset.
        * random_state: Random seed for shuffling the dataset.
        ### RETURN
        * Resultant HF Dataset containing 'input','output',and 'task' column.
        """
        data = []
        if multiply:
            # Extraction
            for task in tasks["extraction"]:
                data.extend(self.create_intermediate_data(self.data,"extraction",[task],incomplete_target_format={}))
            # Imputation
            for task in tasks["imputation"].keys():
                data.extend(self.create_intermediate_data(self.data,"imputation",[task],incomplete_target_format=tasks["imputation"]))
        else:
            # Extraction
            data.extend(self.create_intermediate_data(self.data,"extraction",tasks["extraction"],incomplete_target_format=[]))
            # Imputation
            data.extend(self.create_intermediate_data(self.data,"imputation",tasks["imputation"].keys().tolist(),incomplete_target_format=tasks["imputation"]))
        
        for i_row, row in enumerate(data):
            data[i_row] = self.stringify_input_output(row,self.prompter,self.pattern,self.prompt_side)

        # Turn interim data into HF Dataset
        data = pd.DataFrame(data)
        if shuffle:
            data = data.sample(frac=1.0,random_state=random_state)
        data = Dataset.from_pandas(data)

        return data
    
    def build_test_data(self,task:str="acos",paradigm:str="extraction",incomplete_targets:List[List[Dict]]=None) -> Dataset:
        """
        ### DESC
            Method for building the test HF Dataset.
        ### PARAMS
        * task: The task.
        * paradigm: extraction or imputation.
        * incomplete_targets: List of list of dictionary target (Used for imputation).
        ### RETURN
        * Resultant HF Dataset containing 'input','output',and 'task' column.
        """
        if incomplete_targets != None:
            assert len(self.data) == len(incomplete_targets)
        test_data = []
        for i_row, row in enumerate(self.data):
            targets = self.reduce_targets(row["target"],task)
            targets = handle_mix_sentiment(targets)
            targets = remove_duplicate_targets(targets)
            row = {
                "text" : row["text"],
                "target" : targets,
                "incomplete_target" : None,
                "paradigm" : paradigm,
                "task" : task
            }
            if incomplete_targets != None:
                if len(incomplete_targets) == len(self.data):
                    incomplete_targets[i_row] = handle_mix_sentiment(incomplete_targets[i_row])
                    incomplete_targets[i_row] = remove_duplicate_targets(incomplete_targets[i_row])
                    row["incomplete_target"]= incomplete_targets[i_row]
            row = self.stringify_input_output(row,self.prompter,self.pattern,self.prompt_side)
            row["target"] = targets
            test_data.append(row)
        test_data = Dataset.from_pandas(pd.DataFrame(test_data))
        return test_data
    
    def create_intermediate_data(self,data:List[Dict],paradigm:str="extraction",task:List=["aos"],incomplete_target_format:Dict[str,List]=["ao"]) -> List[Dict]:
        """
        ### DESC
            Method creating intermediate dataset.
        ### PARAMS
        * data: Dataset.
        * paradigm: Paradigm of the ABSA task (extraction or imputation).
        * task: List of task names, if more than one task, every row will get a task retrieved from a round robin manner from the list. Example: ao, aos, as, acos, etc.
        * incomplete_target_format: List of incomplete target format, if more than one format, every imputation row will get a incomplete format retrieved from a random manner from the list. Example: ao, aos, as, acos, etc.
        ### RETURN
        * result_data: Resulting intermediate dataset.
        """
        result_data = []
        for i_row, row in enumerate(data):
            chosen_task = task[i_row%len(task)]
            targets = self.reduce_targets(row["target"],chosen_task)
            targets = handle_mix_sentiment(targets)
            targets = remove_duplicate_targets(targets)
            incomplete_targets = None
            if len(incomplete_target_format.keys()) > 0:
                chosen_incomplete_target_format = random.choice(incomplete_target_format[chosen_task])
                incomplete_targets = self.reduce_targets(targets,chosen_incomplete_target_format)
            result_row = {
                "text" : row["text"],
                "target" : targets,
                "incomplete_target" : incomplete_targets,
                "paradigm" : paradigm,
                "task" : chosen_task
            }
            result_data.append(result_row)
        return result_data

    def stringify_input_output(self,row:Dict,prompter:Prompter=Prompter(),pattern:Pattern=Pattern(),prompt_side:str="left") -> Dict:
        """
        ### DESC
            Mapping method for creating input output (designed for Huggingface trainer).
        ### PARAMS
        * row: Data row.
        * prompter: Prompter object.
        * pattern: Pattern object.
        * prompt_side: Prompt side, either 'left' or 'right'.
        ### RETURN
        * Dictionary containing input and output.
        """
        text = row["text"]
        targets = eval(row["target"]) if isinstance(row["target"],str) else row["target"]
        incomplete_targets = eval(row["incomplete_target"])  if isinstance(row["incomplete_target"],str) else row["incomplete_target"]
        # Masking the special chars
        text = pattern.masking(text)
        for i_target in range(len(targets)):
            for key, value in targets[i_target].items():
                targets[i_target][key] = pattern.masking(value)
        if incomplete_targets != None:
            for i_incomplete_target in range(len(incomplete_targets)):
                for key, value in incomplete_targets[i_incomplete_target].items():
                    incomplete_targets[i_incomplete_target][key] = pattern.masking(value)
        # build prompt
        prompt = prompter.build_prompt(row["task"],pattern,incomplete_targets,row["paradigm"])
        input_text = prompt + ' ' + text if prompt_side == "left" else text + ' ' + prompt
        output_text = pattern.batch_stringify(targets,row["task"])

        return {"input" : input_text, "output" : output_text, "task" : row["task"]}
    
    def process_num_targets(self,text:str,num_targets:List[tuple],target_format:str) -> List[Dict]:
        """
        ### DESC
            Method for processing num targets to target in the format list of dictionaries.
        ### PARAMS
        * text: Text source.
        * num_targets: Targets in the form list of tuples, may consist of aspect term or opinion term indexes.
        * target_format: The target format. Example: acos, aos, ac, ao, etc.
        ### RETURN
        * result_targets: The resultant targets in the form list of dictionaries.
        """
        splitted_text = text.split()
        result_targets = []
        for num_target in num_targets:
            assert len(num_target) == len(target_format) # number of element in the num targets must be the same with the task
            target = {}
            for i, se in enumerate(target_format): # iterate a, c, o, s
                assert se in 'acos'
                key = SENTIMENT_ELEMENT[se]
                if se == 'a' or se == 'o':
                    if num_target[i] != [-1]: # Implicit aspect
                        value = ' '.join([splitted_text[j] for j in num_target[i]])
                    else:
                        value = IMPLICIT_ASPECT
                elif se == 's':
                    value = SENTTAG2WORD[num_target[i]]
                else: # se == 'c
                    value = num_target[i]
                target[key] = value
            result_targets.append(target)
        return result_targets
    
    def read_data(self,path:str,target_format:str="aos") -> List[Dict]:
        f""""
        ### DESC
            Method to read dataset. Each line is in the format of TEXT{SEP}TARGETS .
        ### PARAMS
        * path: Data path.
        ### RETURN
        * data: List of dictionaries.
        """
        assert path.endswith(".txt")
        with open(path,'r') as reader:
            data = reader.read().strip().splitlines()
        for i,line in enumerate(data):
            try:
                text, num_targets = line.split(SEP)
                num_targets = eval(num_targets)
                targets = self.process_num_targets(text,num_targets,target_format)
            except Exception as e:
                raise ValueError(f"Each line should be in the format 'TEXT{SEP}TARGET'. Example: {sample}")
            data[i] = {"text" : text, "target" : targets}
        return data
    
    def reduce_targets(self,targets:List[Dict],task:str="ao") -> List[Dict]:
        """
        ### DESC
            Method to reduce sentiment elements in the designated targets.
        ### PARAMS
        * targets: ABSA targets containing sentiment elements.
        * task: The task related to the resulting target.
        ### RETURN
        * result_targets: The resultant targets.
        """
        result_targets = []
        for target in targets:
            result_target = target.copy()
            for se in "acos":
                key = SENTIMENT_ELEMENT[se]
                if se not in task and key in result_target:
                    del result_target[key]
            result_targets.append(result_target)
        return result_targets

class NonABSADataset(CustomDataset):
    """
    Non-ABSA Dataset.
    """
    def __init__(self,data_path:str,prompt_side:str="left"):
        """
        ### DESC
            Constructor for NonABSADataset instance.
        ### PARAMS
        * data_path: Path to the dataset file (in csv format).
        * prompt_side: "left" or "right" (prompt placement).
        """
        super().__init__()
        assert data_path.endswith(".csv")
        self.data = pd.read_csv(data_path)
        assert "text" in self.data.columns and "output" in self.data.columns and "prompt" in self.data.columns
        self.data_path = data_path
        self.prompt_side = prompt_side
    
    def build_data(self) -> Dataset:
        """
        ### DESC
            Method for building HF Dataset.
        ### RETURN
        * Resultant HF Dataset containing 'input','output',and 'task' column.
        """
        result_data = self.data.copy()
        result_data["input"] = result_data.apply(lambda row: self.add_prompt(row.prompt,row.text,self.prompt_side),axis=1)
        result_data = result_data[["input","output"]]
        result_data["task"] = ["non_absa" for _ in range(len(self.data))]
        result_data = Dataset.from_pandas(result_data)
        return result_data
    
    def add_prompt(self,prompt:str,text:str,prompt_side:str="left") -> str:
        """
        ### DESC
            Method to add prompt.
        ### PARAMS
        * prompt: The prompt.
        * text: The text.
        * prompt_side: "left" or "right" (prompt placement).
        ### RETURN
        * Prompted text.
        """
        assert prompt_side == "left" or prompt_side == "right"
        return prompt + ": " + text if prompt_side == "left" else text + prompt + ": "

if __name__ == "__main__":
    data_path = "./sample_dataset.txt"
    target_format = "aos"
    tasks = {
        "extraction" : ["as","aos","ao"],
        "imputation" : {
            "aos" : ["ao","as"]
        }
    }
    pattern = Pattern(tasks=["ao","as","aos","os","a"],
                      categories=["LAPTOP#GENERAL","BATTERY#HEALTH"])
    prompter = Prompter()
    prompt_side = "left"

    absa_ds = ABSADataset(data_path=data_path,
                          target_format=target_format,
                          prompter=prompter,
                          prompt_side=prompt_side,
                          pattern=pattern)
    
    train = absa_ds.build_train_val_data(tasks=tasks,multiply=True,shuffle=False,random_state=0)
    test1 = absa_ds.build_test_data(task="ao",paradigm="extraction",incomplete_targets=None)
    test2 = absa_ds.build_test_data(task="ao",paradigm="imputation",incomplete_targets=[[{"aspect" : "kocak"}] for _ in range(len(absa_ds))])
    train.to_csv("train.csv",index=False)
    test1.to_csv("extraction.csv",index=False)
    test2.to_csv("imputation.csv",index=False)

    non_absa = NonABSADataset("../data/doc_sa/en/kaggle/interim/amazonReview.csv","left").build_data()
    non_absa.to_csv("non_absa.csv",index=False)
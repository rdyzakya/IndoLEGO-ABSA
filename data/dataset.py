from datasets import Dataset
from pattern import Pattern
from prompt import Prompter
import pandas as pd
import random

from typing import Dict, List

from constant import SEP, SENTTAG2WORD, SENTIMENT_ELEMENT

# THINGS TO DO
# a. Consider tasks
# b. Consider prompter
# c. Consider random shuffle
# d. Consider extraction & imputation
# e. Multiply the data or not

sample = "It rarely works and when it does it 's incredibly slow .####[([2], [1], 'NEG')]"

class ABSADataset:
    """
    ABSA Dataset.
    """
    def __init__(self,data_path:str,target_format:str="acos",tasks:Dict={"extraction" : ["ao","ac","as"],"imputation" : {"acos" : ["ao","ac","aos"]}},prompter:Prompter=Prompter(),prompt_side:str="left",pattern:Pattern=Pattern(),multiply:bool=True,multiply_imputation:bool=False,shuffle:bool=True,random_state:int=None):
        """
        ### DESC
            Constructor method for ABSA dataset.
        ### PARAMS
        * data_path: Path to ABSA txt dataset.
        * target_format: Format of the targets in the dataset file. Example: ao, aosc, aos, etc.
        * tasks: Dictionary for available tasks. Available keys are 'extraction' and 'imputation'. Task names indicates the sentiment elements to be extracted, available sentiment elements are Aspect Term (a or A), Opinion Term (o or O), Sentiment (s or S), and Category (c or C). Example of task name: aoc, aos, aosc, ao, etc.
        * prompter: Prompter object to add prompt.
        * prompt_side: Prompt side, either 'left' or 'right'.
        * pattern: Pattern object.
        * multiply: Multiply the dataset (True) or randomly assign random task to the data (uniform distribution).
        * multiply_imputation: Multiply the dataset row for each incomplete target task. Example: 'acos' imputed with the incomplete target gained from 'ac' or 'ao' or 'as' or etc. If multiply then one row become multiplied as three row.
        * shuffle: Shuffle the dataset.
        * random_state: Seed for randomize the shuffle (only works when shuffle equals True).
        """
        super().__init__()
        # Assert data type
        assert isinstance(data_path,str)
        assert isinstance(target_format,str)
        assert isinstance(tasks,dict)
        assert isinstance(prompter,Prompter)
        assert isinstance(prompt_side,str)
        assert isinstance(pattern,Pattern)
        assert isinstance(multiply,bool)
        assert isinstance(multiply,bool)
        assert isinstance(multiply_imputation,bool)
        assert isinstance(shuffle,bool)
        assert isinstance(random_state,int) or random_state == None

        # Assert key of tasks (paradigms)
        tasks_key = set(tasks.keys())
        assert tasks_key.issubset({"extraction", "imputation"})
        # Assert prompt side
        assert prompt_side in ["left","right"]

        # Read the data
        data = self.read_data(data_path)
        
        new_data = []
        categories = []
        for row_index, row in enumerate(data):
            text, num_targets = row["text"], row["num_targets"]
            original_targets = self.process_num_targets(text,num_targets,task=target_format)
            
            # Record categories for category related tasks.
            for target in original_targets:
                if "category" in target.keys():
                    if target["category"] not in categories:
                        categories.append(target["category"])
            
            # Multiply
            if multiply:
                for task in tasks["extraction"]:
                    target = self.reduce_target(original_targets,task)
                    new_data_entry = {
                        "text" : text,
                        "paradigm" : "extraction",
                        "task" : task,
                        "target" : target,
                        "incomplete_target" : None
                    }
                    new_data.append(new_data_entry)
                for main_task, incomplete_task_list in tasks["imputation"].items():
                    if multiply_imputation:
                        for incomplete_target_task in incomplete_task_list:
                            target = self.reduce_target(original_targets,main_task)
                            incomplete_target = self.reduce_target(original_targets,incomplete_target_task)
                            new_data_entry = {
                                "text" : text,
                                "paradigm" : "imputation",
                                "task" : task,
                                "target" : target,
                                "incomplete_target" : incomplete_target
                            }
                            new_data.append(new_data_entry)
                    else:
                        incomplete_target_task = random.choice(incomplete_task_list)
                        target = self.reduce_target(original_targets,main_task)
                        incomplete_target = self.reduce_target(original_targets,incomplete_target_task)
                        new_data_entry = {
                            "text" : text,
                            "paradigm" : "imputation",
                            "task" : task,
                            "target" : target,
                            "incomplete_target" : incomplete_target
                        }
                        new_data.append(new_data_entry)
            else: # Not multiply
                extraction_paradigm_task_tuple = [("extraction",extraction_task) for extraction_task in tasks["extraction"]]
                imputation_paradigm_task_tuple = [("imputation",imputation_task) for imputation_task in tasks["imputation"].keys()]
                paradigm_task_tuple = extraction_paradigm_task_tuple + imputation_paradigm_task_tuple

                # round robin manner
                chosen_paradigm_task_tuple = paradigm_task_tuple[row_index%len(paradigm_task_tuple)]
                paradigm, task = chosen_paradigm_task_tuple
                target = self.reduce_target(original_targets,task)
                if paradigm == "extraction":
                    new_data_entry = {
                        "text" : text,
                        "paradigm" : paradigm,
                        "task" : task,
                        "target" : target,
                        "incomplete_target" : None
                    }
                    new_data.append(new_data_entry)
                else:
                    incomplete_task_list = tasks["imputation"][task]
                    if multiply_imputation:
                        for incomplete_target_task in incomplete_task_list:
                            incomplete_target = self.reduce_target(original_targets,incomplete_target_task)
                            new_data_entry = {
                                "text" : text,
                                "paradigm" : "imputation",
                                "task" : task,
                                "target" : target,
                                "incomplete_target" : incomplete_target
                            }
                            new_data.append(new_data_entry)
                    else:
                        incomplete_target_task = random.choice(incomplete_task_list)
                        incomplete_target = self.reduce_target(original_targets,incomplete_target_task)
                        new_data_entry = {
                            "text" : text,
                            "paradigm" : "imputation",
                            "task" : task,
                            "target" : target,
                            "incomplete_target" : incomplete_target
                        }
                        new_data.append(new_data_entry)
        
        # Shuffle
        if shuffle:
            if isinstance(random_state,int):
                def seed():
                    return random_state
                random.shuffle(new_data,seed)
            else:
                random.shuffle(new_data)
        # Assign data_frame and dataset attribute
        self.data_frame = pd.DataFrame(new_data)
        self.dataset = Dataset.from_pandas(self.data_frame).map(self.create_input_output)
    
    def create_input_output(self,row:Dict,prompter:Prompter=Prompter(),pattern:Pattern=Pattern(),prompt_side:str="left") -> Dict:
        """
        ### DESC
            Mapping method for creating input output (designed for uggingface trainer).
        ### PARAMS
        * row: Data row.
        * prompter: Prompter object.
        * pattern: Pattern object.
        * prompt_side: Prompt side, either 'left' or 'right'.
        ### RETURN
        * Dictionary containing input and output.
        """
        text = row["text"]
        prompt = prompter.build_prompt(row["task"],pattern,row["incomplete_target"],row["paradigm"])
        input_text = prompt + ' ' + text if prompt_side == "left" else text + ' ' + prompt
        output_text = f" {pattern.inter_sep} ".join([pattern.stringify(target,row["task"]) for target in row["target"]])# pattern.stringify(row["target"],row["task"])

        return {"input" : input_text, "output" : output_text}
    
    def process_num_targets(self,text:str,num_targets:List[tuple],task:str) -> List[Dict]:
        """
        ### DESC
            Method for processing num targets to target in the format list of dictionaries.
        ### PARAMS
        * text: Text source.
        * num_targets: Targets in the form list of tuples, may consist of aspect term or opinion term indexes.
        * task: The designated task. Example: axos, aos, ac, ao, etc.
        ### RETURN
        * result_targets: The resultant targets in the form list of dictionaries.
        """
        splitted_text = text.split()
        result_targets = []
        for num_target in num_targets:
            assert len(num_target) == len(task) # number of element in the num targets must be the same with the task
            target = {}
            for i, se in enumerate(task): # iterate a, c, o, s
                assert se in 'acos'
                key = SENTIMENT_ELEMENT[se]
                if se == 'a' or se == 'o':
                    value = ' '.join([splitted_text[j] for j in num_target[i]])
                elif se == 's':
                    value = SENTTAG2WORD[num_target[i]]
                else: # se == 'c
                    value = num_target[i]
                target[key] = value
            result_targets.append(target)
        return result_targets
    
    def read_data(self,path:str) -> List[Dict]:
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
            except Exception as e:
                raise ValueError(f"Each line should be in the format 'TEXT{SEP}TARGET'. Example: {sample}")
            data[i] = {"text" : text, "num_targets" : num_targets}
        return data
    
    def reduce_target(self,target:Dict[str,str],task:str="ao") -> Dict[str,str]:
        """
        ### DESC
            Method to reduce sentiment elements in the designated targets.
        ### PARAMS
        * target: An ABSA target containing sentiment elements.
        * task: The task related to the resulting target.
        ### RETURN
        * result_target: The resultant target.
        """
        result_target = target.copy()
        for se in "acos":
            key = SENTIMENT_ELEMENT[se]
            if se not in task and key in result_target:
                del result_target[key]
        return result_target


if __name__ == "__main__":
    data_path = "./sample_dataset.txt"
    target_format = "aos"
    tasks = {
        "extraction" : ["ao", "as", "aos", "os"],
        "imputation" : {
            "aos" : ["ao", "as", "a"]
        }
    }
    pattern = Pattern(task=["ao","as","aos","os","a"],
                      categories=["LAPTOP#GENERAL","BATTERY#HEALTH"])
    prompter = Prompter()
    prompt_side = "left"

    absa_ds = ABSADataset(data_path=data_path,
                          target_format=target_format,
                          tasks=tasks,
                          pattern=pattern,
                          prompter=prompter,
                          prompt_side=prompt_side)
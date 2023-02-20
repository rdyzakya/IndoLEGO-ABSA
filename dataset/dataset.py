from datasets import Dataset
from pattern import Pattern
from prompt import Prompter
import pandas as pd
import random

from typing import Dict, List

from constant import SEP, SENTTAG2WORD, SENTIMENT_ELEMENT, IMPLICIT_ASPECT

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
            for original_target in original_targets:
                if "category" in original_target.keys():
                    if original_target["category"] not in categories:
                        categories.append(original_target["category"])
            
            # Multiply
            if multiply:
                for task in tasks["extraction"]:
                    targets = [self.reduce_target(original_target,task) for original_target in original_targets]
                    new_data_entry = {
                        "text" : text,
                        "paradigm" : "extraction",
                        "task" : task,
                        "target" : targets,
                        "incomplete_target" : None
                    }
                    new_data.append(new_data_entry)
                for main_task, incomplete_task_list in tasks["imputation"].items():
                    if multiply_imputation:
                        for incomplete_target_task in incomplete_task_list:
                            targets = [self.reduce_target(original_target,main_task) for original_target in original_targets]
                            incomplete_targets = [self.reduce_target(original_target,incomplete_target_task) for original_target in original_targets]
                            if len(targets) > 0:
                                new_data_entry = {
                                    "text" : text,
                                    "paradigm" : "imputation",
                                    "task" : main_task,
                                    "target" : targets,
                                    "incomplete_target" : incomplete_targets
                                }
                                new_data.append(new_data_entry)
                    else:
                        incomplete_target_task = random.choice(incomplete_task_list)
                        targets = [self.reduce_target(original_target,main_task) for original_target in original_targets]
                        incomplete_targets = [self.reduce_target(original_target,incomplete_target_task) for original_target in original_targets]
                        if len(targets) > 0:
                            new_data_entry = {
                                "text" : text,
                                "paradigm" : "imputation",
                                "task" : main_task,
                                "target" : targets,
                                "incomplete_target" : incomplete_targets
                            }
                            new_data.append(new_data_entry)
            else: # Not multiply
                extraction_paradigm_task_tuple = [("extraction",extraction_task) for extraction_task in tasks["extraction"]]
                imputation_paradigm_task_tuple = [("imputation",imputation_task) for imputation_task in tasks["imputation"].keys()]
                paradigm_task_tuple = extraction_paradigm_task_tuple + imputation_paradigm_task_tuple

                # round robin manner
                chosen_paradigm_task_tuple = paradigm_task_tuple[row_index%len(paradigm_task_tuple)]
                paradigm, task = chosen_paradigm_task_tuple
                targets = [self.reduce_target(original_target,task) for original_target in original_targets]
                if paradigm == "extraction":
                    new_data_entry = {
                        "text" : text,
                        "paradigm" : paradigm,
                        "task" : task,
                        "target" : targets,
                        "incomplete_target" : None
                    }
                    new_data.append(new_data_entry)
                else:
                    incomplete_task_list = tasks["imputation"][task]
                    if multiply_imputation:
                        for incomplete_target_task in incomplete_task_list:
                            incomplete_targets = [self.reduce_target(original_target,incomplete_target_task) for original_target in original_targets]
                            if len(targets) > 0:
                                new_data_entry = {
                                    "text" : text,
                                    "paradigm" : "imputation",
                                    "task" : task,
                                    "target" : targets,
                                    "incomplete_target" : incomplete_targets
                                }
                                new_data.append(new_data_entry)
                    else:
                        incomplete_target_task = random.choice(incomplete_task_list)
                        incomplete_targets = [self.reduce_target(original_target,incomplete_target_task) for original_target in original_targets]
                        if len(targets) > 0:
                            new_data_entry = {
                                "text" : text,
                                "paradigm" : "imputation",
                                "task" : task,
                                "target" : targets,
                                "incomplete_target" : incomplete_targets
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
        # Preprocess targets (for mixed sentiments) and also remove duplicates
        for i in range(len(new_data)):
            targets = new_data[i]["target"]
            incomplete_targets = new_data[i]["incomplete_target"]
            targets = self.preprocess_targets(targets)
            targets = self.remove_duplicate_targets(targets)
            if incomplete_targets != None:
                incomplete_targets = self.preprocess_targets(incomplete_targets)
                incomplete_targets = self.remove_duplicate_targets(incomplete_targets)
            new_data[i]["target"] = targets
            new_data[i]["incomplete_target"] = incomplete_targets
        # Assign data_frame and dataset attribute
        self.data_frame = pd.DataFrame(new_data)
        # Make the target and incomplete target as a string first, because HF Dataset tends to make all the dictionary having the same keys.
        self.data_frame["target"] =  self.data_frame["target"].astype(str)
        self.data_frame["incomplete_target"] =  self.data_frame["incomplete_target"].astype(str)
        self.dataset = Dataset.from_pandas(self.data_frame)
        # Change back the target fromn string to dictionary in create input output
        self.dataset = self.dataset.map(lambda x : self.create_input_output(x,prompter,pattern,prompt_side),remove_columns=self.dataset.column_names)
    
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
        targets = eval(row["target"])
        incomplete_targets = eval(row["incomplete_target"])
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
    
    def preprocess_targets(self,targets:List[Dict]) -> List[Dict]: ### MAY CONTAIN BUG, AFFECTS ORDER
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
                    target["sentiment"] = sentiments[0]
                    results_targets.append(target)
                while non_sentiment_target in non_sentiment_target_stack:
                    non_sentiment_target_index = non_sentiment_target_stack.index(non_sentiment_target)
                    non_sentiment_target_stack.pop(non_sentiment_target_index)
                    sentiment_target_stack.pop(non_sentiment_target_index)
        return results_targets
    
    def remove_duplicate_targets(self,targets:List[Dict]) -> List[Dict]:
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

class NonABSADataset:
    """
    Non-ABSA Dataset.
    """
    def __init__(self,data_path:str):
        """
        ### DESC
            Constructor for NonABSADataset instance.
        ### PARAMS
        * data_path: Path to the dataset file (in csv format).
        """
        assert data_path.endswith(".csv")
        self.data_frame = pd.read_csv(data_path)
        assert "input" in self.data_frame.columns and "output" in self.data_frame.columns
        self.dataset = Dataset.from_pandas(self.data_frame)

if __name__ == "__main__":
    data_path = "./sample_dataset.txt"
    target_format = "aos"
    tasks = {
        "extraction" : ["as","aos","ao"],
        "imputation" : {
            "aos" : ["ao","as"]
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
    absa_ds.dataset.to_csv("dataset_result.csv",index=False)
    absa_ds.data_frame.to_csv("data_frame_result.csv",index=False)
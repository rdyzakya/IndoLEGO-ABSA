import random
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict
from .prompter import Prompter
from .ans_constructor import AnswerConstructor

available_algo = ["random", "round_robin"]

class DataAugmentator:
    """
    Responsible to conduct data augmentation.
    """
    def __init__(self):
        self.prompter = Prompter()
        self.ans_constructor = AnswerConstructor()
    
    def do(self, data:List[Dict], nt_se_order:str, tasks:List[Dict], n_fold:int=1, algo:str="round_robin", shuffle=True) -> List[Dict]:
        """
        ### DESC
            Method to conduct data augmentation.
        ### PARAMS
        * data: Data created from DataReader instance.
        * nt_se_order: Sentiment order in the num_targets' tuples, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * tasks: List of dictionary containing the task details. See "task_example" method to know the tasks example.
        * n_fold: The amount of dataset that is multiplied.
        * algo: Task sampling algorithm.
        * shuffle: If shuffle is true, then the data will be shuffled using random library.
        ### RETURN
        * List of dictionary containing augmented dataset.
        """
        result = []
        pbar = tqdm(total=n_fold*len(data))

        for i in range(len(data)):
            for n in range(n_fold):
                row = data[i]
                chosen_task = None
                if algo == "round_robin":
                    chosen_task = deepcopy(tasks[((i * n_fold) + n)%len(tasks)])
                elif algo == "random":
                    chosen_task = deepcopy(random.choice(tasks))
                else:
                    raise NotImplementedError(f"Available task sampling algorithm: {available_algo}")
                
                prompt_method = chosen_task.pop("prompt")
                answer_method = chosen_task.pop("answer")

                # prompter
                prompt_func = getattr(self.prompter, prompt_method)
                # answer
                ans_func = getattr(self.ans_constructor, answer_method)

                prompt_args = deepcopy(chosen_task)
                prompt_args["text"] = row["text"]

                # input
                inputs = prompt_func(**prompt_args)
                # output
                out = ans_func(text=row["text"], 
                            num_targets=row["num_targets"], 
                            nt_se_order=nt_se_order, 
                            se_order=chosen_task["se_order"])
                result_row = {
                    "input" : inputs,
                    "output" : out,
                    "se_order" : chosen_task["se_order"]
                }
                result.append(result_row)
                pbar.update(1)

        if shuffle:
            random.shuffle(result)
        return result
    
    def __call__(self, data:List[Dict], nt_se_order:str, tasks:List[Dict], n_fold:int, algo:str, shuffle=True) -> List[Dict]:
        """
        ### DESC
            Conduct data augmentation.
        ### PARAMS
        * data: Data created from DataReader instance.
        * nt_se_order: Sentiment order in the num_targets' tuples, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * tasks: List of dictionary containing the task details. See "task_example" method to know the tasks example.
        * n_fold: The amount of dataset that is multiplied.
        * algo: Task sampling algorithm.
        * shuffle: If shuffle is true, then the data will be shuffled using random library.
        ### RETURN
        * List of dictionary containing augmented dataset.
        """
        return self.do(data=data,
                       nt_se_order=nt_se_order,
                       tasks=tasks,
                       n_fold=n_fold,
                       algo=algo,
                       shuffle=shuffle)
    
    def task_example(self) -> List[Dict]:
        """
        Example of tasks value in the "do" or call method.
        """
        return [
            {
                "se_order" : "aos",
                "prompt" : "lego_absa",
                "answer" : "lego_absa"
            },
            {
                "se_order" : "ao",
                "prompt" : "lego_absa",
                "answer" : "lego_absa"
            },
            {
                "se_order" : "as",
                "prompt" : "lego_absa",
                "answer" : "lego_absa"
            },
            {
                "se_order" : 'a',
                "prompt" : "lego_absa",
                "answer" : "lego_absa"
            },
            {
                "se_order" : 'o',
                "prompt" : "lego_absa",
                "answer" : "lego_absa"
            }
        ]
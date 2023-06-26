# 🤗
from .dataset import *
from .answer import *
from .prompt import *
import random
from copy import deepcopy
from tqdm import tqdm

sample_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "aos",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "ao",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "imputation",
        "reduced_se_order" : "ao",
        "se_order" : "aos",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "fewshot",
        "se_order" : "aos",
        "method" : "lego_absa",
        "n_shot" : 2
    }
]

ans_const = AnswerConstructor()
prompter = {
    "extraction" : ExtractPrompter(),
    "imputation" : ImputePrompter(),
    "fewshot" : FewShotPrompter()
}

def data_gen(data, nt_se_order, tasks, n_fold, algo):
    result = []
    pbar = tqdm(total=n_fold*len(data))
    for n in range(n_fold):
        for i in range(len(data)):
            el = data[i]
            chosen_task = None
            if algo == "round_robin":
                chosen_task = deepcopy(tasks[i%len(tasks)])
                if (chosen_task["paradigm"] == "imputation") and \
                (chosen_task["paradigm"] == "fewshot") and \
                (len(el["num_targets"]) == 0):
                    pbar.update(1)
                    continue
            elif algo == "random":
                chosen_task = deepcopy(random.choice(tasks))
                while (chosen_task["paradigm"] == "imputation") and \
                (chosen_task["paradigm"] == "fewshot") and \
                (len(el["num_targets"]) == 0):
                    chosen_task = deepcopy(random.choice(tasks))
            else:
                raise NotImplementedError
            paradigm = chosen_task.pop("paradigm")
            method = chosen_task.pop("method")

            # prompter
            prompt_func = getattr(prompter[paradigm], method)
            # answer
            ans_func = getattr(ans_const, method)

            prompt_args = deepcopy(chosen_task)
            if paradigm == "extraction":
                prompt_args["text"] = el["text"]
            elif paradigm == "imputation" or paradigm == "fewshot":
                prompt_args["text"] = el["text"]
                prompt_args["num_targets"] = el["num_targets"]
                prompt_args["nt_se_order"] = nt_se_order

            # input
            inputs = prompt_func(**prompt_args)
            # output
            out = ans_func(text=el["text"], 
                           num_targets=el["num_targets"], 
                           nt_se_order=nt_se_order, 
                           se_order=chosen_task["se_order"])
            result.append({
                "input" : inputs,
                "output" : out,
                "se_order" : chosen_task["se_order"]
            })
            pbar.update(1)
    return result
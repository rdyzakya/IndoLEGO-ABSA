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
        "prompt" : "lego_absa",
        "answer" : "lego_absa"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "ao",
        "prompt" : "lego_absa",
        "answer" : "lego_absa"
    },
    {
        "paradigm" : "imputation",
        "reduced_se_order" : "ao",
        "se_order" : "aos",
        "prompt" : "lego_absa",
        "answer" : "lego_absa"
    },
    {
        "paradigm" : "fewshot",
        "se_order" : "aos",
        "prompt" : "lego_absa",
        "answer" : "lego_absa",
        "n_shot" : 2
    },
    {
        "paradigm" : "denoise",
        "se_order" : "aos",
        "method" : "lego_absa"
    }
]

ans_const = AnswerConstructor()
prompter = {
    "extraction" : ExtractPrompter(),
    "imputation" : ImputePrompter(),
    "fewshot" : FewShotPrompter(),
    "denoise" : DenoisingPrompter()
}

def data_gen(data, nt_se_order, tasks, n_fold, algo, shuffle=True):
    result = []
    pbar = tqdm(total=n_fold*len(data))
    for i in range(len(data)):
        for n in range(n_fold):
            el = data[i]
            chosen_task = None
            if algo == "round_robin":
                chosen_task = deepcopy(tasks[((i * n_fold) + n)%len(tasks)])
                if chosen_task["paradigm"] in ["imputation","denoise"] and len(el["num_targets"]) == 0:
                    pbar.update(1)
                    continue
            elif algo == "random":
                chosen_task = deepcopy(random.choice(tasks))
                while chosen_task["paradigm"] in ["imputation","denoise"] and len(el["num_targets"]) == 0:
                    chosen_task = deepcopy(random.choice(tasks))
            else:
                raise NotImplementedError
            paradigm = chosen_task.pop("paradigm")
            prompt_method = chosen_task.pop("prompt")
            answer_method = chosen_task.pop("answer")

            # prompter
            prompt_func = getattr(prompter[paradigm], prompt_method)
            # answer
            ans_func = getattr(ans_const, answer_method)

            prompt_args = deepcopy(chosen_task)
            if paradigm == "extraction":
                prompt_args["text"] = el["text"]
            elif paradigm == "imputation":
                prompt_args["text"] = el["text"]

                num_targets = el["num_targets"]
                reduced_se_order = prompt_args.pop("reduced_se_order")
                reduced_num_targets = reduce_num_targets(num_targets, nt_se_order, reduced_se_order)
                reduced_targets = process_num_targets(text=el["text"], num_targets=reduced_num_targets, se_order=reduced_se_order)
                
                prompt_args["reduced_targets"] = reduced_targets
                # prompt_args["num_targets"] = el["num_targets"]
                # prompt_args["nt_se_order"] = nt_se_order
            elif paradigm == "fewshot":
                prompt_args["text"] = el["text"]
                targets = process_num_targets(text=el["text"], num_targets=el["num_targets"], se_order=nt_se_order)
                prompt_args["targets"] = targets
            elif paradigm == "denoise":
                prompt_args["text"] = el["text"]
                targets = process_num_targets(text=el["text"], num_targets=el["num_targets"], se_order=nt_se_order)
                prompt_args["targets"] = targets
            else:
                raise NotImplementedError

            # input
            inputs = prompt_func(**prompt_args)
            # output
            out = ans_func(text=el["text"], 
                           num_targets=el["num_targets"], 
                           nt_se_order=nt_se_order, 
                           se_order=chosen_task["se_order"])
            row = {
                "input" : inputs,
                "output" : out,
                "se_order" : chosen_task["se_order"],
                "paradigm" : paradigm
            }
            if row not in result:
                result.append(row)
            pbar.update(1)
    if shuffle:
        random.shuffle(result)
    return result
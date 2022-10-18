import json
import re
import random
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
pattern_token = {
    "aspect" : "<A>",
    "opinion" : "<O>",
    "sentiment" : "<S>"
}
available_task = ["ate","ote","aste","aope","uabsa"]
special_char = {
    "[" : "\[",
    "]" : "\]",
    "." : "\.",
    "\\" : "\\",
    "{" : "\{",
    "}" : "\}",
    "^" : "\^",
    "$" : "\$",
    "*" : "\*",
    "+" : "\+",
    "?" : "\?",
    "|" : "\|",
    "(" : "\(",
    ")" : "\)"
}

class Prompter:
    def __init__(self,prompt_path,prompt_side="left"):
        # with open(prompt_path,'r',encoding="utf-8") as f:
        #     self.prompts = [el.strip() for el in f.readlines()]
        # self.prompts = prompts
        self.prompts = json.load(open(prompt_path,'r',encoding="utf-8"))
        self.available_option = list(range(len(self.prompts))) + ["random"]
        if prompt_side != "left" and prompt_side != "right":
            raise ValueError(f"Prompt side should only be 'left' or 'right'")
        self.prompt_side = prompt_side
        # self.option = option
    
    def compile(self,prompt_side,option):
        if option not in self.available_option:
            raise ValueError(f"Option should only be from {self.available_option}")
        if prompt_side != "left" and prompt_side != "right":
            raise ValueError(f"Prompt side should only be 'left' or 'right'")
        self.prompt_side = prompt_side
        self.option = option
    
    def add_prompt(self,task,texts,option):
        if option not in self.available_option:
            raise ValueError(f"Option should only be from {self.available_option}")
        prompt_side = self.prompt_side
        chosen_prompt = None
        prompts = []
        if isinstance(option,int):
            chosen_prompt = self.prompts[task][option]
            prompts = [chosen_prompt for _ in texts]
        result = []
        for t in texts:
            if option == "random":
                chosen_prompt = random.choice(self.prompts[task])
                prompts.append(chosen_prompt)
            t = chosen_prompt + ' ' + t if prompt_side == "left" else t + ' ' + chosen_prompt
            result.append(t)
        return result, prompts

class Pattern:
    def __init__(self,task,open_bracket='(',close_bracket=')',intra_sep='|',inter_sep=','):
        self.task = task.split()
        self.open_bracket = open_bracket.strip()
        self.close_bracket = close_bracket.strip()
        self.intra_sep = intra_sep.strip()
        self.inter_sep = inter_sep.strip()

        self.pattern = {}
        for t in self.task:
            if t not in available_task:
                raise ValueError(f"Task only can be from {available_task}, combine it by using spaces. Your task(s) : {task}")
            self.pattern[t] = []
            if t != "ote":
                self.pattern[t].append(pattern_token["aspect"])
            if t == "ote" or t == "aope" or t == "aste":
                self.pattern[t].append(pattern_token["opinion"])
            if t == "uabsa" or t =="aste":
                self.pattern[t].append(pattern_token["sentiment"])
            # print(t)
            self.pattern[t] = f"{open_bracket} " + f" {intra_sep.strip()} ".join(self.pattern[t]) + f" {close_bracket}"
            self.pattern[t] = self.pattern[t].strip()
    
    def regex(self,task):
        regex_pattern = self.pattern[task]
        intra_sep = self.intra_sep
        for k,v in special_char.items():
            regex_pattern = regex_pattern.replace(k,v)
            intra_sep = intra_sep.replace(k,v)
        for k,v in pattern_token.items():
            if k == "sentiment":
                regex_pattern = regex_pattern.replace(v,f"(?P<sentiment>{'|'.join(senttag2word.values())})")
            else:
                regex_pattern = regex_pattern.replace(v,f"(?P<{k}>[^{intra_sep}]+)")
        regex_pattern = regex_pattern.replace(' ',r'\s*')
        return regex_pattern

    def stringify(self,dict_instance,task):
        result = self.pattern[task]
        for k in dict_instance.keys():
            result = result.replace(pattern_token[k],dict_instance[k])
        return result
    
    def __repr__(self):
        return str(self.pattern)
    
    def __str__(self):
        return str(self.pattern)

available_paradigm = ["extraction","annotation"]

def process_numtargets(text, num_target, task="aste"):
    if task not in available_task:
        raise NotImplementedError
    sent = text.split()
    res = []
    for tup in num_target:
        aspect_index, opinion_index, sentiment = tup
        sentiment = senttag2word[sentiment]
        aspect = ' '.join([sent[aspect_index[i]] for i in range(len(aspect_index))])
        opinion = ' '.join([sent[opinion_index[i]] for i in range(len(opinion_index))])
        instance = {}
        if task == "ate" or task =="aste" or task =="uabsa" or task =="aope":
            instance["aspect"] = aspect
        if task == "ote" or task == "aste" or task =="aope":
            instance["opinion"] = opinion
        if task == "aste" or task =="uabsa":
            instance["sentiment"] = sentiment
        if instance not in res: # not duplicating
            res.append(instance)
    return res

no_target = "NONE"

def stringify_target(text, num_target, target, task, paradigm="extraction", pattern=Pattern(task=' '.join(available_task))):
    if paradigm == "extraction":
        stringified_target = []
        target = eval(target)
        if len(target) == 0:
            return no_target
        for t in target:
            st = pattern.stringify(t,task)
            stringified_target.append(st)
        return f" {pattern.inter_sep} ".join(stringified_target)
    else:
        raise NotImplementedError

def batch_stringify_target(batch_text, batch_num_target, batch_target, batch_task, paradigm="extraction", pattern=Pattern(task=' '.join(available_task))):
  res = [stringify_target(text, num_target, target, task, paradigm, pattern) for text, num_target, target, task in zip(batch_text, batch_num_target, batch_target, batch_task)]
  return res

def inverse_stringify_target(stringified_target, task, paradigm="extraction", pattern=Pattern(task=' '.join(available_task))):
    if stringified_target.strip() == '' or stringified_target.strip() == no_target:
        return []
    if paradigm != "extraction":
        raise NotImplementedError
    regex_pattern = pattern.regex(task=task)
    inverse_stringified_targets = [i.groupdict() for i in re.finditer(regex_pattern,stringified_target)]
    for i in range(len(inverse_stringified_targets)):
        for k,v in inverse_stringified_targets[i].items():
            inverse_stringified_targets[i][k] = v.strip()
    return inverse_stringified_targets

def batch_inverse_stringify_target(batch_stringified_target, batch_task, paradigm="extraction", pattern=Pattern(task=' '.join(available_task))):
    res = [inverse_stringify_target(stringified_target=stringified_target, 
        task=task, paradigm=paradigm, pattern=pattern) for stringified_target,task in zip(batch_stringified_target,batch_task)]
    return res

def post_process_lm_result(texts_without_target,texts_with_target,tokenizer,encoding_args,decoding_args):
    assert len(texts_without_target) == len(texts_with_target)
    
    encoded_texts_without_target = tokenizer.batch_encode_plus(texts_without_target,**encoding_args)
    decoded_texts_without_target = tokenizer.batch_decode(encoded_texts_without_target["input_ids"],**decoding_args)
    
    result = [
        t1[len(t2):].strip() for t1, t2 in zip(texts_with_target,decoded_texts_without_target)
    ]

    return result
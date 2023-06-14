from typing import Dict, List

SENTTAG2WORD = {"POS": "positive", "NEG": "negative", "NEU": "neutral", "MIX" : "mixed"}
SENTIMENT_ELEMENT = {'a' : "aspect", 'o' : "opinion", 's' : "sentiment", 'c' : "category"}
SEP = "####"
NO_TARGET = "NONE"
IMPLICIT_ASPECT = "NULL"

sample = f"It rarely works and when it does it 's incredibly slow .{SEP}[([2], [1], 'NEG')]"

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

def process_num_targets(text:str,num_targets:List[tuple],target_format:str) -> List[Dict]:
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

def read_data(path:str,target_format:str="aos") -> List[Dict]:
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
    unique_categories = []
    for i,line in enumerate(data):
        try:
            text, num_targets = line.split(SEP)
            num_targets = eval(num_targets)
            targets = process_num_targets(text,num_targets,target_format)
            for target in targets:
                if "category" in target.keys():
                    unique_categories.append(target["category"])
        except Exception as e:
            raise ValueError(f"Each line should be in the format 'TEXT{SEP}TARGET' and format {target_format}. Example: {sample}. Yours: {line}")
        data[i] = {"text" : text, "target" : targets, "num_targets" : num_targets}
    return data

def reduce_targets(targets:List[Dict],task:str="ao") -> List[Dict]:
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
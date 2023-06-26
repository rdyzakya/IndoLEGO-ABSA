import sys
sys.path.append("..")
import constant
from typing import Dict, List, Tuple

sample = f"It rarely works and when it does it 's incredibly slow .{constant.SEP}[([2], [1], 'NEG')]"

def handle_mix_sentiment(targets:List[Dict]) -> List[Dict]: ### MAY CONTAIN BUG, AFFECTS ORDER
    """
    ### DESC
        Method to preprocess targets (especially to reduce target containing sentiments).
    ### 
    * targets: List of targets.
    ### RETURN
    * Resulting targets.
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

def read_data(path:str) -> List[Dict]:
    f""""
    ### DESC
        Method to read dataset. Each line is in the format of TEXT{constant.SEP}TARGETS .
    ### PARAMS
    * path: Data path.
    ### RETURN
    * List of dictionaries.
    """
    assert path.endswith(".txt")
    with open(path,'r') as reader:
        data = reader.read().strip().splitlines()
    for i,line in enumerate(data):
        try:
            text, num_targets = line.split(constant.SEP)
        except Exception as e:
            raise ValueError(f"Each line should be in the format 'TEXT{constant.SEP}TARGET'. Example: {sample}. Yours: {line}")
        num_targets = eval(num_targets)
        data[i] = {"text" : text, "num_targets" : num_targets}
    return data
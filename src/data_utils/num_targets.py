import sys
sys.path.append("..")
import constant
from typing import Dict, List, Tuple

def process_num_targets(text:str, num_targets:List[Tuple], se_order:str) -> List[Dict]:
    """
    ### DESC
        Method for processing num targets to target in the format list of dictionaries.
    ### PARAMS
    * text: Text source.
    * num_targets: Targets in the form list of tuples, may consist of aspect term or opinion term indexes.
    * se_order: The sentiment element order. Example: acos, aos, ac, ao, etc.
    ### RETURN
    * The resultant targets in the form list of dictionaries.
    """
    splitted_text = text.split()
    result_targets = []
    for num_target in num_targets:
        assert len(num_target) == len(se_order) # number of element in the num targets must be the same with the task
        target = {}
        for i, se in enumerate(se_order): # iterate a, c, o, s
            assert se in constant.SENTIMENT_ELEMENT
            key = constant.SENTIMENT_ELEMENT[se]
            if se == 'a' or se == 'o':
                if num_target[i] != [-1]: # Implicit aspect
                    value = ' '.join([splitted_text[j] for j in num_target[i]])
                else:
                    value = constant.IMPLICIT_ASPECT
            elif se == 's':
                value = constant.SENTTAG2WORD[num_target[i]]
            else: # se == 'c
                value = num_target[i]
            target[key] = value
        result_targets.append(target)
    return result_targets

def remove_duplicate(arr:List) -> List:
    """
    ### DESC
        Method for removing duplicates in list.
    ### PARAMS
    * targets: List.
    ### RETURN
    * Result list.
    """
    result = []
    for el in arr:
        if el not in result:
            result.append(el)
    return result

def reduce_num_targets(num_targets:List[Tuple], src_se_order:str="aos", tgt_se_order:str="ao") -> List[Tuple]:
    """
    ### DESC
        Reduce num_target to fewer sentiment elements.
    ### PARAMS
    * num_targets: List of num_target.
    * src_se_order: Sentiment element order from the source num_targets.
    * tgt_se_order: Sentiment element order for the target num_targets.
    ### RETURN
    * Reduced num_targets.
    """
    assert set(src_se_order).issubset(constant.SENTIMENT_ELEMENT)
    assert set(tgt_se_order).issubset(src_se_order)
    if len(num_targets) > 0:
        assert len(src_se_order) == len(num_targets[0])
    
    tgt_index = []
    for se in tgt_se_order:
        index = src_se_order.index(se)
        tgt_index.append(index)
    
    result = []
    for nt in num_targets:
        reduced_nt = []
        for index in tgt_index:
            reduced_nt.append(nt[index])
        reduced_nt = tuple(reduced_nt)
        result.append(reduced_nt)

    result = remove_duplicate(result)

    return result
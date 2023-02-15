from typing import List, Dict

"""
The metrics in this research follows the following rule:
1. The definition of true positive is said that an ABSA tuple that exist in the prediction list, also exist in the target list
2. The definition of the false positive is said that an ABSA tuple that exist in the prediction list do not exist in the target list
3. The definition of the false negative is said that an ABSA tuple that exist in the target list do not exist in the prediction list
"""

def recall(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        Recall metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * Recall value.
    """
    true_positive = 0
    false_negative = 0
    for prediction,target in zip(predictions,targets):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
    return true_positive/(true_positive + false_negative)

def precision(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        Precision metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * Precision value.
    """
    true_positive = 0
    false_positive = 0
    for prediction,target in zip(predictions,targets):
        for prediction_tuple in prediction:
            if prediction_tuple in target:
                true_positive += 1
            else:
                false_positive += 1
    return true_positive/(true_positive + false_positive)

def f1_score(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        F1 score metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * F1 score value.
    """
    recall_value = recall(predictions,targets)
    precision_value = precision(predictions,targets)
    return (2 * recall_value * precision_value)/(recall_value + precision_value)
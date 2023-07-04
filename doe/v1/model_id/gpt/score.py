import sys
import json
import re
sys.path.append("../../../src/")
from evaluation import recall, precision, f1_score, summary_score
import data_utils

with open("william_hotel.json",'r') as fp:
    pred = json.load(fp)

def catch_answer(output,se_order):
    if output == "NULL":
        return []
    # output = output.replace("<pad>",'')
    # output = output.replace("</s>",'')
    pattern = []
    for se in se_order:
        if se != 's':
            pattern.append(f"\s*(?P<{data_utils.SENTIMENT_ELEMENT[se]}>[^;]+)\s*")
        else:
            pattern.append(f"\s*(?P<{data_utils.SENTIMENT_ELEMENT['s']}>positive|negative|neutral)\s*")
    pattern = ','.join(pattern)
    pattern = f"\({pattern}\)"
    found = [found_iter.groupdict() for found_iter in re.finditer(pattern,output)]
    for i in range(len(found)):
        for k, v in found[i].items():
            found[i][k] = found[i][k].strip()
    return found


str_pred = [el["str_pred"].split("( opinion, aspect, sentiment )")[-1].strip() for el in pred]
preds = [catch_answer(el,"oas") for el in str_pred]
labels = [el["target"] for el in pred]

score = summary_score(preds,labels)

with open("score.json",'w') as fp:
    json.dump(score,fp)
# print(preds[:5])
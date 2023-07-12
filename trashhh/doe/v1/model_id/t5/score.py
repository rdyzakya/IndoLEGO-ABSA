import sys
import json
sys.path.append("../../../src/")
from evaluation import recall, precision, f1_score, summary_score

with open("william_hotel.json",'r') as fp:
    pred = json.load(fp)

preds = [el["pred"] for el in pred]
labels = [el["target"] for el in pred]

score = summary_score(preds,labels)

with open("score.json",'w') as fp:
    json.dump(score,fp)
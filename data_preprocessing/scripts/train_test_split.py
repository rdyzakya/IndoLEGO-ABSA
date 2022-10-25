from sklearn.model_selection import train_test_split
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='filename of txt', required=True)
parser.add_argument('-o', '--output_path', help='output path', required=True)
parser.add_argument('-r', '--random_state', type=int, help='random state', required=False)
parser.add_argument('-ts', '--test_size', type=float, help='test size', default=0.2)
args = parser.parse_args()

random_seed = args.random_state

path = args.file

if not path.endswith(".txt"):
    path += ".txt"

test_frac = args.test_size
f = open(path,'r',encoding="utf-8")
data = f.readlines()
f.close()
data = [el.strip() for el in data]

train, test = train_test_split(data,test_size=test_frac,random_state=random_seed)

with open(os.path.join(args.output_path,"train.txt"),'w',encoding="utf-8") as writer:
    for line in train:
        writer.write(line + "\n")

with open(os.path.join(args.output_path,"test.txt"),'w',encoding="utf-8") as writer:
    for line in test:
        writer.write(line + "\n")
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--foldername', help='foldername', required=True)
args = parser.parse_args()

def rename(path):
    splitted = os.path.split(path)
    fname = splitted[-1]
    end = "." + fname.split(".")[-1]
    fname = fname.replace(end,"")
    new_fname = fname.split("_")[0] + end
    result = [el for el in splitted[:-1]] + [new_fname]
    return os.path.join(*result)

listdir = os.listdir(args.foldername)

for fname in listdir:
    src = os.path.join(args.foldername,fname)
    os.rename(src,rename(src))
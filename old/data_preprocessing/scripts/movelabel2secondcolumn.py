import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--foldername', help='foldername of datasaur tsv', required=True)
parser.add_argument('-o', '--output_path', help='output path', required=True)
args = parser.parse_args()

all_files = [el for el in os.listdir(args.foldername) if el.endswith(".tsv") and not el.endswith("-reformat.tsv")]
base_path = os.getcwd()

for fname in all_files:
    with open(os.path.join(args.foldername,fname),'r',encoding="utf-8") as writer:
        lines = writer.readlines()
        for i,line in enumerate(lines):
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            splitted_line = line.split()
            # switch to the second column
            splitted_line[3], splitted_line[4] = splitted_line[4], splitted_line[3]
            lines[i] = '\t'.join(splitted_line) + '\n'
    with open(os.path.join(args.output_path,fname[:-4] + "-reformat.tsv"),'w',encoding="utf-8") as writer:
        for line in lines:
            writer.write(line)
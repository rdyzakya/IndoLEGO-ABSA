import pandas as pd
import re
import os

import argparse

def comparator(x,df_):
    df_contained_x = df_.loc[df_.label.str.contains(x,regex=False)]
    index = df_contained_x.iloc[0].name
    length = df_contained_x.shape[0]
    return index, -length

# parse foldername of datasaur tsv
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--foldername', help='foldername of datasaur tsv', required=True)
parser.add_argument('-o', '--output', help='output filename', required=True)
args = parser.parse_args()
foldername = args.foldername
output = args.output

# verify if output filename is txt
if output.endswith('.txt'):
    output = output
else:
    # change the format of output filename to txt
    output = output + '.txt'

filenames = os.listdir(foldername)

df = pd.DataFrame()

for filename in filenames:
    with open(foldername + '/' + filename, 'r') as f:
        data = f.readlines()
        data = [line for line in data if line != "\n" and not line.startswith('#')]
        data_dict = []
        for line in data:
            splitted_line = line.split('\t')
            sentence_id = splitted_line[0].split('-')[0]
            token_id = splitted_line[0].split('-')[1]
            start_pos = splitted_line[1].split('-')[0]
            end_pos = splitted_line[1].split('-')[1]
            token = splitted_line[2]
            label = splitted_line[4]
            if label != "_":
                splitted_label = label.split('|')
                splitted_label = [el for el in splitted_label if el.startswith('concept') or el.startswith('positive') or el.startswith('negative')]
                label = '|'.join(splitted_label)
            new_el = {
                'sentence_id': sentence_id,
                'token_id': token_id,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'token': token,
                'label': label
            }
            data_dict.append(new_el)
        data_df = pd.DataFrame(data_dict)
        data_df['BIO_concept'] = ""
        data_df['BIO_sm'] = ""
        for index,row in data_df.iterrows():
            label = row.label
            if label == "_":
                data_df.loc[index,'BIO_concept'] = "O"
                data_df.loc[index,'BIO_sm'] = "O"
            else:
                splitted_label = label.split("|")
                splitted_label = sorted(splitted_label, key=lambda x:comparator(x,data_df))

                BIO_concept_label = ""
                BIO_sm_label = ""

                for l in splitted_label:
                    # concept (choose the longer one as the B)
                    if l.startswith("concept"):
                        if index == 0:
                            BIO_concept_label = "B-CONCEPT"
                        elif data_df.loc[index-1,'BIO_concept'] == "O":
                                BIO_concept_label = "B"
                        else:
                            BIO_concept_label = "I-CONCEPT"
                
                    # sentiment marker (choose the longer one as the B)
                    if l.startswith("positive") or l.startswith("negative"):
                        if index == 0:
                            BIO_sm_label = "B"
                        elif data_df.loc[index-1,'BIO_sm'] == "O":
                                BIO_sm_label = "B"
                        else:
                            BIO_sm_label = "I"
                    if l.startswith("positive") and not re.search("-POS",l):
                        BIO_sm_label += "-POS"
                    if l.startswith("negative") and not re.search("-NEG",l):
                        BIO_sm_label += "-NEG"
                data_df.loc[index,'BIO_concept'] = BIO_concept_label
                data_df.loc[index,'BIO_sm'] = BIO_sm_label
        df = df.append(data_df)

concept_output_filename = output.replace('.txt','_concept.txt')
with open(concept_output_filename,"w") as f:
    for _,row in df.iterrows():
        if row.token_id == "1" and row.sentence_id != "1":
            f.write("\n")
        f.write(f"{row['token']}\t{row['BIO_concept']}\n")

sm_output_filename = output.replace('.txt','_sm.txt')
with open(sm_output_filename,"w") as f:
    for _,row in df.iterrows():
        if row.token_id == "1" and row.sentence_id != "1":
            f.write("\n")
        f.write(f"{row['token']}\t{row['BIO_sm']}\n")
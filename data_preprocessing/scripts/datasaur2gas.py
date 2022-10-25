import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--foldername', help='foldername of datasaur tsv', required=True)
parser.add_argument('-o', '--output', help='output filename', required=True)
args = parser.parse_args()
foldername = args.foldername
output = args.output

if not output.endswith('.txt'):
    output += '.txt'

filenames = os.listdir(foldername)

filenames = [filename for filename in filenames if filename.endswith('.tsv')]

all_df = pd.DataFrame()

for filename in filenames:
    with open(foldername + '/' + filename, 'r', encoding='utf-8') as f:
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
            relation = []
            if label != "_":
                splitted_label = label.split('|')
                for l in splitted_label:
                    if not (l.startswith('concept') or l.startswith('positive') or l.startswith('negative')):
                        relation.append(l)
                        # remove from splitted_label
                        splitted_label.remove(l)
                label = '|'.join(splitted_label)
            if len(relation) > 0:
                relation = '|'.join(relation)
            else:
                relation = '_'
            new_el = {
                'sentence_id': sentence_id,
                'token_id': token_id,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'token': token,
                'label': label,
                'relation': relation
            }
            data_dict.append(new_el)
        data_df = pd.DataFrame(data_dict)
        sentence_ids = data_df.sentence_id.unique()

        aste_data = []
        for sentence_id in sentence_ids:
            sentence_data = data_df.loc[data_df.sentence_id == sentence_id]
            tokens = sentence_data.token.values.tolist()
            text = " ".join(tokens)

            labels = sentence_data.label.unique().tolist()
            if "_" in labels:
                labels.remove("_")
            labels = "|".join(labels).split("|")

            concept_labels = [l for l in labels if l.startswith('concept')]
            sm_labels = [l for l in labels if l.startswith('positive') or l.startswith('negative') or l.startswith('neutral')]

            concept_labels = [(concept_label, concept_label.split("[")[1].split("]")[0]) for concept_label in concept_labels]
            sm_labels = [(sm_label, sm_label.split("[")[1].split("]")[0]) for sm_label in sm_labels]

            concept_labels = sorted(list(set(concept_labels)), key=lambda x: x[1])
            sm_labels = sorted(list(set(sm_labels)), key=lambda x: x[1])

            relations = data_df.loc[data_df.sentence_id == sentence_id,'relation'].unique().tolist()
            if "_" in relations:
                relations.remove("_")
            relations = "|".join(relations).split("|")

            current_sentence_data = {"text":text,"triplets":[]}

            for concept_label,i in concept_labels:
                for sm_label,j in sm_labels:
                    connection = f"[{i}_{j}]"
                    concept_token_index = [int(token_id)-1 for token_id in sentence_data.token_id[sentence_data.label.str.contains(concept_label,regex=False)].values.tolist()]
                    sm_token_index = [int(token_id)-1 for token_id in sentence_data.token_id[sentence_data.label.str.contains(sm_label,regex=False)].values.tolist()]
                    triplet = (concept_token_index, sm_token_index, sm_label[:3].upper())
                    if connection in relations:
                        current_sentence_data["triplets"].append(triplet)
            # if len(current_sentence_data["triplets"]) > 0: # new
            aste_data.append(current_sentence_data)
        relation_df = pd.DataFrame(aste_data)
        all_df = all_df.append(relation_df)
with open(output, 'w', encoding='utf-8') as f:
    for index, row in all_df.iterrows():
        f.write(row.text + "####" + str(row.triplets) + "\n")
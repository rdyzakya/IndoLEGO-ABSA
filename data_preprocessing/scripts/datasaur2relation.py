import pandas as pd
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

if not output.endswith(".csv"):
    output += ".csv"

filenames = os.listdir(foldername)

filenames = [filename for filename in filenames if filename.endswith('.tsv')]

all_df = pd.DataFrame()

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

        relation_data = []
        for sentence_id in sentence_ids:
            tokens = data_df.loc[data_df.sentence_id == sentence_id, 'token'].values.tolist()
            text = " ".join(tokens)

            labels = data_df.loc[data_df.sentence_id == sentence_id,'label'].unique().tolist()
            if '_' in labels:
                labels.remove("_")
            labels = "|".join(labels).split("|")

            concept_labels = [l for l in labels if l.startswith('concept')]
            sm_labels = [l for l in labels if l.startswith('positive') or l.startswith('negative')]

            concept_labels_id = [int(x.split("[")[1].split("]")[0]) for x in concept_labels]
            sm_labels_id = [int(x.split("[")[1].split("]")[0]) for x in sm_labels]

            concept_labels_id = sorted(list(set(concept_labels_id)))
            sm_labels_id = sorted(list(set(sm_labels_id)))

            relations = data_df.loc[data_df.sentence_id == sentence_id,'relation'].unique().tolist()
            relations.remove("_")
            relations = "|".join(relations).split("|")

            for i in concept_labels_id:
                for j in sm_labels_id:
                    concept_tokens = data_df.loc[(data_df.sentence_id == sentence_id) & data_df.label.str.contains("concept[" + str(i) + "]",regex=False), 'token'].values.tolist()
                    sm_tokens = data_df.loc[(data_df.sentence_id == sentence_id) & (data_df.label.str.contains("positive[" + str(j) + "]",regex=False) | data_df.label.str.contains("negative[" + str(j) + "]",regex=False)), 'token'].values.tolist()
                    concept_text = " ".join(concept_tokens)
                    sm_text = " ".join(sm_tokens)
                    new_relation_data_el = {
                        "text_a" : text,
                        "text_b" : f"{concept_text}-{sm_text}"
                    }
                    connection = f"[{i}_{j}]"
                    if connection in relations:
                        new_relation_data_el["label"] = 1
                    else:
                        new_relation_data_el["label"] = 0
                    relation_data.append(new_relation_data_el)
        relation_df = pd.DataFrame(relation_data)
        os.makedirs(foldername + "/relation_data", exist_ok=True)
        # append relation_df to all_df
        all_df = all_df.append(relation_df)
        relation_df.to_csv(os.path.join(foldername,"relation_data",filename.replace(".tsv",".csv")), index=False)
all_df.to_csv(output, index=False)
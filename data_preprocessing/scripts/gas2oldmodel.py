import argparse
import os
import pandas as pd

senttag2word = {"POS" : "positive", "NEG" : "negative", "NEU" : "neutral"}

# parse arguments (file path, etc.)
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file path", type=str, required=True)
parser.add_argument("--output", help="output file path", default="output", required=True)
args = parser.parse_args()

concept_output = args.output + "_concept.txt"
sentiment_marker_output = args.output + "_sentiment_marker.txt"
relation_output = args.output + "_relation.csv"

f = open(args.file, "r")
data = f.read()
f.close()
data = data.strip().split('\n')

sep = "####"

for i in range(len(data)):
    line = data[i]
    splitted_line = line.split(sep)
    text = splitted_line[0]
    splitted_text = text.split(' ')
    label = eval(splitted_line[1])

    data[i] = {
        "text": splitted_text,
        "label": label,
    }

concept_file = open(concept_output, "w", encoding="utf-8")
sentiment_marker_file = open(sentiment_marker_output, "w", encoding="utf-8")
relation_df = pd.DataFrame(columns=["text_a", "text_b", "label"])
for el in data:
    # concept and sentiment marker
    splitted_text = el["text"]
    label = el["label"]
    for i in range(len(splitted_text)):
        concept_file.write(splitted_text[i] + "\t")
        sentiment_marker_file.write(splitted_text[i] + "\t")
        concept_token = "O"
        sentiment_marker_token = "O"
        for triplet in label:
            concept_index = triplet[0]
            sentiment_marker_index = triplet[1]
            polarity = triplet[2]
            polarity_relation = senttag2word[polarity]

            if i == concept_index[0]:
                concept_token = "B-CONCEPT" # -" + polarity
            elif i in concept_index:
                concept_token = "I-CONCEPT" # -" + polarity
            
            if i == sentiment_marker_index[0]:
                sentiment_marker_token = "B-" + polarity
            elif i in sentiment_marker_index:
                sentiment_marker_token = "I-" + polarity
        concept_file.write(concept_token + "\n")
        sentiment_marker_file.write(sentiment_marker_token + "\n")
    concept_file.write("\n")
    sentiment_marker_file.write("\n")

    all_concept = []
    all_sentiment_marker = []

    if len(label) == 0: # tanpa triplet
        relation_df = pd.concat([relation_df, pd.DataFrame({
            "text_a": [" ".join(splitted_text)],
            "text_b": [""],
            "label": ["1"],
        })], ignore_index=True)
    
    for triplet_1 in label:
        for triplet_2 in label:
            concept_1 = triplet_1[0]
            sentiment_marker_2 = triplet_2[1]

            if concept_1 not in all_concept:
                all_concept.append(concept_1)
            if sentiment_marker_2 not in all_sentiment_marker:
                all_sentiment_marker.append(sentiment_marker_2)
    for concept_1 in all_concept:
        for sentiment_marker_2 in all_sentiment_marker:
            combination = [
                (concept_1, sentiment_marker_2,"POS"),
                (concept_1, sentiment_marker_2,"NEG"),
                (concept_1, sentiment_marker_2,"NEU"),
            ]
            joined_sent = " ".join(splitted_text)
            joined_concept = " ".join(splitted_text[concept_1[0]:concept_1[-1]+1])
            joined_sentiment_marker = " ".join(splitted_text[sentiment_marker_2[0]:sentiment_marker_2[-1]+1])
            pair = f"{joined_concept.lower()}-{joined_sentiment_marker.lower()}"
            if combination[0] in label or combination[1] in label or combination[2] in label: # ini memakai index posisi
                relation_df = pd.concat([relation_df, pd.DataFrame({
                    "text_a": [joined_sent],
                    "text_b": [pair],
                    "label": ["1"],
                })], ignore_index=True)
            else:
                relation_df = pd.concat([relation_df, pd.DataFrame({
                    "text_a": [joined_sent],
                    "text_b": [pair],
                    "label": ["0"],
                })], ignore_index=True)

concept_file.close()
sentiment_marker_file.close()

index_to_drop = []

unique_text_a = relation_df.text_a.unique()
for text_a in unique_text_a:
    rows_a = relation_df.loc[relation_df.text_a == text_a]
    unique_text_b = rows_a.text_b.unique()
    for text_b in unique_text_b:
        rows_b = rows_a.loc[rows_a.text_b == text_b]
        if rows_b.shape[0] > 1:
            unique_label = rows_b.label.unique()
            if len(unique_label) > 1: # ada yang 0 dan 1
                indexes = rows_b.loc[rows_b.label == "0"].index.tolist() # hilangkan yang 0
                index_to_drop.extend(indexes)

# drop yang duplicate
# yang 0 dan 1
relation_df = relation_df.drop(index=index_to_drop)
# drop sleuruh yang duplicate
relation_df = relation_df.drop_duplicates()

relation_df.to_csv(relation_output, index=False)
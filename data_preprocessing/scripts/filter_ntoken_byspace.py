import requests

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='filename of txt', required=True)
parser.add_argument('-o', '--output', help='output filename', required=True)
parser.add_argument('-n', '--n_token', type=int, help="Desired n token", default=256)
args = parser.parse_args()

path = args.file

if not path.endswith(".txt"):
    path += ".txt"

if not args.output.endswith(".txt"):
    args.output += ".txt"

tokenizerURL = "http://10.181.131.244:8778/tokenizer"
def tokenize(sentence,type="all",with_index=True):
    r = requests.post(url=tokenizerURL, json={'text': sentence, "version": "v1", "type": type})
    tokenizedSentences = r.json()['sentences']
    if not with_index or type != "all":
        return tokenizedSentences
    
    new_sentence = tokenizedSentences.copy()
    for i,sentence in enumerate(tokenizedSentences):
        new_sentence[i] = " ".join(sentence)
    new_sentence = "  ".join(new_sentence)
    pos = 0
    for idx,sen in enumerate(tokenizedSentences):
        for jdx, token in enumerate(sen):
            while new_sentence[pos:pos+len(token)] != token:
                pos += 1
                if pos >= len(new_sentence):
                    raise Exception("Tokenizer error")
            start = pos
            end = pos + len(token)
            tokenizedSentences[idx][jdx] = (token,start,end)
            pos += len(token)
    return tokenizedSentences

def main():
    f = open(path,'r',encoding="utf-8")
    data = f.readlines()
    f.close()

    new_data = []
    for l in data:
        splitted = l.split("####")
        text = splitted[0]
        text = text.encode('ascii','ignore').decode('utf-8')
        target = eval(splitted[1])
        splitted_text = text.split()
        if len(splitted_text) <= args.n_token:
            new_data.append(l)
        else:
            tokenized_text = tokenize(text,type="sentence",with_index=False)
            flatten_tokenized_text = []
            for sen in tokenized_text:
                flatten_tokenized_text.extend(sen.split())
            if len(splitted_text) != len(flatten_tokenized_text):
                for i in range(max(len(splitted_text),len(flatten_tokenized_text))):
                    if flatten_tokenized_text[i] != splitted_text[i]:
                        print("ERROR!!!")
                        print(text[:50])
                        print(i)
                        print(flatten_tokenized_text[i-10:i+10])
                        print(splitted_text[i-10:i+10])
                        exit()
            tokenized_text = [sen.split() for sen in tokenized_text]
            pointer = 0
            for sen in tokenized_text:
                new_target = []
                for i,tok in enumerate(sen):
                    for tup in target:
                        aspect_index = tup[0]
                        opinion_index = tup[1]
                        polarity = tup[2]
                        if aspect_index[0] == pointer or opinion_index[0] == pointer:
                            delta = pointer - i
                            for ai in range(len(aspect_index)):
                                aspect_index[ai] -= delta
                            for oi in range(len(opinion_index)):
                                opinion_index[oi] -= delta
                            new_tuple = (aspect_index,opinion_index,polarity)
                            if new_tuple not in new_target:
                                new_target.append(new_tuple)
                    pointer += 1
                new_line = f"{' '.join(sen)}####{new_target}\n"
                new_data.append(new_line)

    f = open(args.output,'w',encoding="utf-8")
    for l in new_data:
        f.write(l)
    f.close()

if __name__ == "__main__":
    main()
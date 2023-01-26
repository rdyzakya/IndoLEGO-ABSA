import requests

import argparse

from transformers import T5Tokenizer


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='filename of txt', required=True)
    parser.add_argument('-o', '--output', help='output filename', required=True)
    parser.add_argument('-n', '--n_token', type=int, help="Desired n token", default=256)
    parser.add_argument('-m','--model_name_or_path', type=str, help="tokenizer model name or path (hf)", default="Wikidepia/IndoT5-small")
    args = parser.parse_args()

    if not args.file.endswith(".txt"):
        args.file += ".txt"

    if not args.output.endswith(".txt"):
        args.output += ".txt"
    
    return args

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

def remap_index(indexes,splitted_text,tokenized_text):
    span = []

    first_idx = indexes[0]
    before = splitted_text[:first_idx]
    joined_before = ''.join(before).replace(' ','')

    for idx in indexes:
        span.append(splitted_text[idx])
    span = ' '.join(span)

    tokenized_span = tokenize(span,with_index=False)[0]

    for i in range(0,len(tokenized_text)-len(tokenized_span)+1):
        window = tokenized_text[i:i+len(tokenized_span)]
        if window == tokenized_span and joined_before == ''.join(tokenized_text[:i]).replace(' ',''):
            new_index = [i+n for n in range(len(tokenized_span))]
            return new_index
    raise Exception(f"Error, span '{span}' not found in text '{' '.join(splitted_text)[:100]}..'")

def flatten(tokenized_text):
    # return [token for sen in tokenized_text for token in sen]
    result = []
    for sen in tokenized_text:
        for token in sen:
            splitted_token = token.split()
            for stoken in splitted_token:
                result.append(stoken)
    return result

def backtokenize(text,num_target):
    splitted_text = text.split()
    tokenized_text = flatten(tokenize(text,type="all",with_index=False))
    new_text = ' '.join(tokenized_text)

    if splitted_text == tokenized_text:
        return new_text, num_target

    new_target = []

    for tup in num_target:
        aspect_index = tup[0]
        opinion_index = tup[1]
        polarity = tup[2]

        new_aspect_index = remap_index(aspect_index,splitted_text,tokenized_text)
        new_opinion_index = remap_index(opinion_index,splitted_text,tokenized_text)

        new_target.append((new_aspect_index,new_opinion_index,polarity))
    return new_text, new_target

def reduce_text(text,num_target):
    sentences = tokenize(text,type="sentence",with_index=False)
    len_sens = []
    checkpoint = 0
    for sen in sentences:
        checkpoint += len(sen.split())
        len_sens.append(checkpoint)

    new_targets = [list() for _ in sentences]

    for tup in num_target:
        aspect_index = tup[0]
        opinion_index = tup[1]
        polarity = tup[2]

        first_aspect_index = aspect_index[0]
        last_aspect_index = aspect_index[-1]

        first_opinion_index = opinion_index[0]
        last_opinion_index = opinion_index[-1]

        for cp_i in range(len(len_sens)):
            start = 0 if cp_i == 0 else len_sens[cp_i-1]
            end = len_sens[cp_i]

            is_aspect_here = first_aspect_index >= start and first_aspect_index < end \
                and last_aspect_index >= start and last_aspect_index < end
            is_opinion_here = first_opinion_index >= start and first_opinion_index < end \
                and last_opinion_index >= start and last_opinion_index < end
            
            # if both in here
            if is_aspect_here and is_opinion_here:
                aspect_index = [idx - start for idx in aspect_index]
                opinion_index = [idx - start for idx in opinion_index]
                new_targets[cp_i].append((aspect_index,opinion_index,polarity))
            else:
                raise Exception(f"There is tuple with different sentence | Sentence : {text[:100]}... | target : {(aspect_index,opinion_index)}")
    return sentences, new_targets

def main():

    args = init_args()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    f = open(args.file,'r',encoding="utf-8")
    data = f.readlines()
    f.close()

    new_data = []
    for l in data:
        splitted = l.split("####")
        text = splitted[0]
        text = text.encode('ascii','ignore').decode('utf-8')
        target = eval(splitted[1])
        
        t5_tokenized_text = tokenizer.tokenize(text)

        if len(t5_tokenized_text) > args.n_token:
            try:
                texts, targets = reduce_text(text,target)
                for _text, _target in zip(texts,targets):
                    new_text, new_target = backtokenize(_text,_target)
                    new_line = f"{new_text}####{new_target}"
                    new_data.append(new_line)
            except:
                new_text, new_target = backtokenize(text,target)
                new_line = f"{new_text}####{new_target}"
                new_data.append(new_line)
        else:
            new_text, new_target = backtokenize(text,target)
            new_line = f"{new_text}####{new_target}"
            new_data.append(new_line)
    
    f = open(args.output,'w',encoding="utf-8")
    for l in new_data:
        f.write(l + "\n")
    f.close()

if __name__ == "__main__":
    main()
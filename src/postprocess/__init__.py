import nltk
import numpy as np

def get_index(el, lst):
    return [index for index, value in enumerate(lst) if value == el]

def generate_combinations(lists):
    if len(lists) == 1:
        return [[x] for x in lists[0]]
    else:
        combinations = []
        for x in lists[0]:
            for y in generate_combinations(lists[1:]):
                combinations.append([x] + y)
        return combinations

def norm(phrase, text):
    splitted_phrase = phrase.split()
    splitted_text = text.split()
    splitted_text = ['' for _ in range(len(splitted_phrase)-1)] + splitted_text + ['' for _ in range(len(splitted_phrase)-1)] # padding
    min_dist = np.inf
    result = phrase
    for i in range(0,len(splitted_text)-len(splitted_phrase)+1):
        window = splitted_text[i:i+len(splitted_phrase)]
        new_phrase = ' '.join(window).strip()
        dist = nltk.edit_distance(phrase, new_phrase)
        if dist < min_dist:
            result = new_phrase
            min_dist = dist
    return result
    
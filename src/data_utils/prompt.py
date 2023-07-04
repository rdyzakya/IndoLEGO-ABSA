import sys
sys.path.append("..")
import constant
import random
import string
from .num_targets import process_num_targets, reduce_num_targets

class ExtractPrompter:
    def lego_absa(self, text, se_order):
        prompt = []
        for counter, se in enumerate(se_order):
            prompt.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
        prompt = " ,".join(prompt)
        result = text + "| " + prompt
        return result
    
    def gas(self, text, se_order):
        prompt = []
        for se in se_order:
            prompt.append(constant.SENTIMENT_ELEMENT[se])
        prompt = " , ".join(prompt)
        prompt = f"( {prompt} )"
        masked_text = text
        # mask
        for k, v in constant.GAS_TOKEN.items():
            masked_text = masked_text.replace(k,v)
        result = masked_text + "| " + prompt
        return result
    
    def bartabsa(self, text, se_order):
        prompt = []
        for se in se_order:
            if se == 'o' or se == 'a':
                name = constant.SENTIMENT_ELEMENT[se]
                start_index = name + "_start"
                end_index = name + "_end"
                prompt.append(start_index)
                prompt.append(end_index)
            else:
                prompt.append(constant.SENTIMENT_ELEMENT[se])
        prompt = ",".join(prompt)
        result = text + "| " + prompt
        return result
    
    def prefix(self, text, se_order):
        prompt = []
        for counter, se in enumerate(se_order):
            prompt.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
        prompt = " ,".join(prompt)
        result = f"Extract with format >> {prompt} | " + text
        return result
    
    def one_token(self, text, se_order):
        result = f"<{se_order}> : " + text
        return result
    
    def autoprompt(self, text, se_order):
        raise NotImplementedError

class ImputePrompter:
    def lego_absa(self, text, reduced_targets, se_order):
        prompt = []
        counter = 0
        # reduced_num_targets = reduce_num_targets(num_targets, nt_se_order, reduced_se_order)
        # reduced_targets = process_num_targets(text=text, num_targets=reduced_num_targets, se_order=reduced_se_order)
        for rt in reduced_targets:
            current = []
            for se in se_order:
                # if se not in reduced_se_order:
                if constant.SENTIMENT_ELEMENT[se] not in rt:
                    counter = counter % 100
                    current.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
                    counter += 1
                else:
                    current.append(constant.SENTIMENT_ELEMENT[se] + " : " + rt[constant.SENTIMENT_ELEMENT[se]])
            current = " ,".join(current)
            if current not in prompt: # no duplicate
                prompt.append(current)
        prompt = " ; ".join(prompt)
        result = text + "| " + prompt
        return result

class FewShotPrompter:
    def lego_absa(self, text, targets, se_order, n_shot):
        prompt = []
        # targets = process_num_targets(text=text, num_targets=num_targets, se_order=nt_se_order)
        shots = random.sample(targets, k=min(n_shot,len(targets)))
        for t in shots:
            current = []
            for se in se_order:
                current.append(constant.SENTIMENT_ELEMENT[se] + " : " + t[constant.SENTIMENT_ELEMENT[se]])
            current = " ,".join(current)
            if current not in prompt: # no duplicate
                prompt.append(current)
        counter = 0

        current = []
        for se in se_order:
            counter = counter % 100
            current.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
            counter += 1
        current = " ,".join(current)
        prompt.append(current)

        prompt = " ; ".join(prompt)
        result = text + "| " + prompt
        return result

def get_index(phrase, text):
    splitted_phrase = phrase.split()
    splitted_text = text.split()
    for i in range(0,len(splitted_text)-len(splitted_phrase)+1):
        window = splitted_text[i:i+len(splitted_phrase)]
        if window == splitted_phrase:
            return (i,i+len(splitted_phrase)-1)
    return (-1,-1)

def noise(phrase, text):
    # 1. insert/delete/substitute 1 word (most left, most right, middle)
    # 2. insert/delete/substitute 1 character
    noise_type = random.randint(0,100)
    noise_level = random.randint(0,100)
    if noise_level <= 50: # char
        char_index = random.randint(0,len(phrase))
        if noise_type <= 33: # insert
            return phrase[:char_index] + random.choice(string.ascii_lowercase) + phrase[char_index:]
        elif noise_type <= 67: # delete
            return phrase[:char_index] + phrase[char_index+1:]
        else: # substitute
            return phrase[:char_index] + random.choice(string.ascii_lowercase) + phrase[char_index+1:]
    else: # word
        phrase_index = get_index(phrase, text)
        splitted_text = text.split()
        splitted_phrase = phrase.split()
        if phrase_index == (-1,-1):
            raise NotImplementedError
        if noise_type <= 33: # insert
            # most left, most right
            if phrase_index == (0,len(splitted_text)):
                return phrase
            elif phrase_index[0] == 0:
                return phrase + ' ' + splitted_text[phrase_index[1]+1]
            elif phrase_index[1] == len(splitted_text)-1:
                return splitted_text[phrase_index[0]-1] + ' ' + phrase
            else:
                return random.choice([
                    phrase + ' ' + splitted_text[phrase_index[1]+1],
                    splitted_text[phrase_index[0]-1] + ' ' + phrase
                ])
        elif noise_type <= 67: # delete
            del_index = random.randint(0,len(splitted_phrase))
            if len(splitted_phrase) == 1:
                return phrase
            else:
                return ' '.join(splitted_phrase[:del_index] + splitted_phrase[del_index+1:])
        else: # substitute
            sub_index = random.randint(0,len(splitted_phrase))
            if len(splitted_phrase) == 1:
                return phrase
            else:
                return ' '.join(splitted_phrase[:sub_index] + random.sample(splitted_text,1) + splitted_phrase[sub_index+1:])

class DenoisingPrompter:
    def lego_absa(self, text, targets, se_order):
        prompt = []
        for t in targets:
            current = []
            noised = False
            for se in se_order:
                element = t[constant.SENTIMENT_ELEMENT[se]]
                noise_decision = random.randint(0,100)
                if se in "ao":
                    if noise_decision <= 50 and not noised and element != constant.IMPLICIT_ASPECT:
                        element = noise(element, text)
                        noised = True
                elif se == 's':
                    if noise_decision <= 50 and not noised:
                        sentiments = list(constant.SENTTAG2WORD.values())
                        sentiments.remove(element)
                        element = random.choice(sentiments)
                        noised = True
                current.append(constant.SENTIMENT_ELEMENT[se] + " : " + element)
            current = " ,".join(current)
            if current not in prompt: # no duplicate
                prompt.append(current)
        prompt = " ; ".join(prompt)
        result = text + "| " + prompt
        return result
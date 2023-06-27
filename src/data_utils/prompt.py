import sys
sys.path.append("..")
import constant
import random
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
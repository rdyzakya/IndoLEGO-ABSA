import sys
import re
sys.path.append("..")
import constant
from typing import Dict, List, Tuple
from .num_targets import process_num_targets

class AnswerConstructor:
    def lego_absa(self, text, num_targets, nt_se_order, se_order):
        targets = process_num_targets(text=text, num_targets=num_targets, se_order=nt_se_order)
        if len(targets) == 0:
            return constant.NO_TARGET
        result = []
        counter = 0
        for t in targets:
            constructed_t = ""
            for se in se_order:
                counter = counter % 100
                constructed_t += ' ' + f"<extra_id_{counter}>" + ' ' + t[constant.SENTIMENT_ELEMENT[se]]
                counter += 1
            constructed_t = constructed_t.strip()
            result.append(constructed_t)
        result = " ; ".join(result)
        return result

    def gas(self, text, num_targets, nt_se_order, se_order):
        targets = process_num_targets(text=text, num_targets=num_targets, se_order=nt_se_order)
        if len(targets) == 0:
            return constant.NO_TARGET
        result = []
        for t in targets:
            constructed_t = []
            for se in se_order:
                element = t[constant.SENTIMENT_ELEMENT[se]]
                constructed_t.append(element)
            constructed_t = " , ".join(constructed_t)
            constructed_t = f"( {constructed_t} )"
            result.append(constructed_t)
        result = " ; ".join(result)
        return result

    def bartabsa(self, text, num_targets, nt_se_order, se_order):
        if len(num_targets) == 0:
            return "-1"
        result = []

        tgt_index = []
        for se in se_order:
            index = nt_se_order.index(se)
            tgt_index.append(index)

        for nt in num_targets:
            for ti in tgt_index:
                el = nt[ti]
                if isinstance(el,list):
                    result.append(str(el[0])) # start index
                    result.append(str(el[-1])) # end index
                else:
                    result.append(el)
        return ','.join(result)

class AnswerCatcher:
    def lego_absa(self, output, se_order, text):
        if output == constant.NO_TARGET:
            return []
        pattern = r""
        for se in se_order:
            if se != 's':
                pattern += f"<extra_id_\d+>\s*(?P<{constant.SENTIMENT_ELEMENT[se]}>[^;]+)\s*"
            else:
                pattern += f"<extra_id_\d+>\s*(?P<{constant.SENTIMENT_ELEMENT['s']}>positive|negative|neutral)\s*"
        result = [found_iter.groupdict() for found_iter in re.finditer(pattern,output)]
        for i in range(len(result)):
            for k, v in result[i].items():
                result[i][k] = v.strip()
        return result
    
    def gas(self, output, se_order, text):
        if output == constant.NO_TARGET:
            return []
        pattern = []
        for se in se_order:
            if se != 's':
                pattern.append(f"\s*(?P<{constant.SENTIMENT_ELEMENT[se]}>[^;]+)\s*")
            else:
                pattern.append(f"\s*(?P<{constant.SENTIMENT_ELEMENT['s']}>positive|negative|neutral)\s*")
        pattern = ','.join(pattern)
        pattern = f"\({pattern}\)"
        result = [found_iter.groupdict() for found_iter in re.finditer(pattern,output)]
        for i in range(len(result)):
            for k, v in result[i].items():
                # unmask
                for k2, v2 in constant.GAS_TOKEN.items():
                    v = v.replace(v2,k2)
                result[i][k] = v.strip()
        return result
    
    def bartabsa(self, output, se_order, text):
        splitted_text = text.split()
        if output == "-1":
            return []
        result = []
        splitted_output = output.split(',')
        splitted_output = [el.strip() for el in splitted_output]

        chunk_size = 0
        for se in se_order:
            if se == 'o' or se == 'a':
                chunk_size += 2
            else:
                chunk_size += 1

        chunks = [
            splitted_output[i:i+chunk_size] for i in range(0,len(splitted_output),chunk_size)
        ]

        chunks = [el for el in chunks if len(el) == chunk_size]

        for el in chunks:
            pred = {}
            cnt_index = 0
            is_invalid = False
            for se in se_order:
                if se == 'a' or se == 'o':
                    start_index = el[cnt_index]
                    end_index = el[cnt_index+1]
                    cnt_index += 2

                    try:
                        start_index = int(start_index)
                        end_index = int(end_index)
                        if end_index < start_index:
                            start_index, end_index = end_index, start_index
                        if start_index == -1 or end_index == -1:
                            pred[constant.SENTIMENT_ELEMENT[se]] = constant.IMPLICIT_ASPECT
                        else:
                            word = splitted_text[start_index:end_index+1]
                            word = ' '.join(word)
                            pred[constant.SENTIMENT_ELEMENT[se]] = word
                    except:
                        is_invalid = True
                        break
                elif se == 's':
                    try:
                        sentiment = constant.SENTTAG2WORD[el[cnt_index]]
                        pred[constant.SENTIMENT_ELEMENT['s']] = sentiment
                    except:
                        is_invalid = True
                        pass
                    cnt_index += 1
                else: # c
                    pred[constant.SENTIMENT_ELEMENT[se]] = el[cnt_index]
                    cnt_index += 1
            if not is_invalid:
                result.append(pred)
        return result
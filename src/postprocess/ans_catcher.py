import re
import sys
sys.path.append("..")
import constant
from typing import List, Dict

class AnswerCatcher:
    """
    Responsible to catch the decomposed answer and transform it to list of dictionary.
    """
    def lego_absa(self, out:str, se_order:str, text:str) -> List[Dict]:
        """
        ### DESC
            Transform LEGO-ABSA decomposed answer.
        ### PARAMS
        * out: Decomposed answer.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * text: Input text. 
        ### RETURN
        * Answer.
        """
        if out == constant.NO_TARGET:
            return []
        pattern = r""
        for se in se_order:
            if se != 's':
                pattern += f"<extra_id_\d+>\s*(?P<{constant.SENTIMENT_ELEMENT[se]}>[^;]+)\s*"
            else:
                pattern += f"<extra_id_\d+>\s*(?P<{constant.SENTIMENT_ELEMENT['s']}>positive|negative|neutral)\s*"
        result = [found_iter.groupdict() for found_iter in re.finditer(pattern,out)]
        for i in range(len(result)):
            for k, v in result[i].items():
                result[i][k] = v.strip()
        return result
    
    def gas(self, out:str, se_order:str, text:str) -> List[Dict]:
        """
        ### DESC
            Transform GAS decomposed answer.
        ### PARAMS
        * out: Decomposed answer.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * text: Input text. 
        ### RETURN
        * Answer.
        """
        if out == constant.NO_TARGET:
            return []
        pattern = []
        for se in se_order:
            if se != 's':
                pattern.append(f"\s*(?P<{constant.SENTIMENT_ELEMENT[se]}>[^;]+)\s*")
            else:
                pattern.append(f"\s*(?P<{constant.SENTIMENT_ELEMENT['s']}>positive|negative|neutral)\s*")
        pattern = ','.join(pattern)
        pattern = f"\({pattern}\)"
        result = [found_iter.groupdict() for found_iter in re.finditer(pattern,out)]
        for i in range(len(result)):
            for k, v in result[i].items():
                # unmask
                for k2, v2 in constant.GAS_TOKEN.items():
                    v = v.replace(v2,k2)
                result[i][k] = v.strip()
        return result
    
    def bartabsa(self, out:str, se_order:str, text:str) -> List[Dict]:
        """
        ### DESC
            Transform BARTABSA decomposed answer.
        ### PARAMS
        * out: Decomposed answer.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * text: Input text. 
        ### RETURN
        * Answer.
        """
        splitted_text = text.split()
        if out == "-1":
            return []
        result = []
        splitted_output = out.split(',')
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
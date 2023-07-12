import sys
sys.path.append("..")
import constant
from typing import Tuple
from .num_targets import process_num_targets, reduce_num_targets

class AnswerConstructor:
    """
    Responsible to construct answer in to the decomposed form.
    """
    def lego_absa(self, text:str, num_targets:Tuple, nt_se_order:str="aos", se_order:str="aos") -> str:
        """
        ### DESC
            LEGO-ABSA answer.
        ### PARAMS
        * text: Text.
        * num_targets: Number targets. Example for aocs: [([1],[2],"restaurant_general","positive")].
        * nt_se_order: Sentiment order in the num_targets' tuples, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * se_order: Sentiment element order for the decomposed answer, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Decomposed answer.
        """
        reduced_nt = reduce_num_targets(num_targets=num_targets, 
                                        src_se_order=nt_se_order, 
                                        tgt_se_order=se_order)
        targets = process_num_targets(text=text, 
                                      num_targets=reduced_nt, 
                                      se_order=se_order)
        
        if len(targets) == 0:
            return constant.NO_TARGET
        
        result = []
        counter = 0
        for t in targets:
            constructed_t = ""
            for se in se_order:
                counter = counter % 100 # maximum index of special token in T5 is 99
                constructed_t += ' ' + f"<extra_id_{counter}>" + ' ' + t[constant.SENTIMENT_ELEMENT[se]]
                counter += 1
            constructed_t = constructed_t.strip()
            result.append(constructed_t)
        result = " ; ".join(result)
        return result

    def gas(self, text:str, num_targets:Tuple, nt_se_order:str="aos", se_order:str="aos") -> str:
        """
        ### DESC
            GAS answer.
        ### PARAMS
        * text: Text.
        * num_targets: Number targets. Example for aocs: [([1],[2],"restaurant_general","positive")].
        * nt_se_order: Sentiment order in the num_targets' tuples, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * se_order: Sentiment element order for the decomposed answer, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Decomposed answer.
        """
        reduced_nt = reduce_num_targets(num_targets=num_targets, 
                                        src_se_order=nt_se_order, 
                                        tgt_se_order=se_order)
        targets = process_num_targets(text=text, 
                                      num_targets=reduced_nt, 
                                      se_order=se_order)
        
        if len(targets) == 0:
            return constant.NO_TARGET
        
        result = []
        for t in targets:
            constructed_t = []
            for se in se_order:
                element = t[constant.SENTIMENT_ELEMENT[se]]

                masked_element = element
                # mask
                for k, v in constant.GAS_TOKEN.items():
                    masked_element = masked_element.replace(k,v)
                    
                constructed_t.append(masked_element)
            constructed_t = " , ".join(constructed_t)
            constructed_t = f"( {constructed_t} )"
            result.append(constructed_t)
        result = " ; ".join(result)
        return result

    def bartabsa(self, text:str, num_targets:Tuple, nt_se_order:str="aos", se_order:str="aos") -> str:
        """
        ### DESC
            BARTABSA answer.
        ### PARAMS
        * text: Text.
        * num_targets: Number targets. Example for aocs: [([1],[2],"restaurant_general","positive")].
        * nt_se_order: Sentiment order in the num_targets' tuples, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        * se_order: Sentiment element order for the decomposed answer, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Decomposed answer.
        """
        reduced_nt = reduce_num_targets(num_targets=num_targets, 
                                        src_se_order=nt_se_order, 
                                        tgt_se_order=se_order)
        
        if len(reduced_nt) == 0:
            return "-1"
        
        result = []

        tgt_index = []
        for se in se_order:
            index = nt_se_order.index(se)
            tgt_index.append(index)

        for nt in reduced_nt:
            for ti in tgt_index:
                el = nt[ti]
                if isinstance(el,list):
                    result.append(str(el[0])) # start index
                    result.append(str(el[-1])) # end index
                else:
                    result.append(el)
        return ','.join(result)
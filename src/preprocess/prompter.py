import sys
sys.path.append("..")
import constant

class Prompter:
    """
    Responsible to add prompt to a text.
    """
    def lego_absa(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            LEGO-ABSA prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
        prompt = []
        for counter, se in enumerate(se_order):
            prompt.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
        prompt = " ,".join(prompt)
        result = text + "| " + prompt
        return result
    
    def gas(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            GAS prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
        prompt = []
        for se in se_order:
            prompt.append(constant.SENTIMENT_ELEMENT[se])
        prompt = " , ".join(prompt)
        prompt = f"( {prompt} )"
        masked_text = text
        result = masked_text + "| " + prompt
        return result
    
    def bartabsa(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            BARTABSA prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
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
    
    def prefix(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            Prefix prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
        prompt = []
        for counter, se in enumerate(se_order):
            prompt.append(constant.SENTIMENT_ELEMENT[se] + " : " + f"<extra_id_{counter}>")
        prompt = " ,".join(prompt)
        result = f"Ekstrak ABSA dengan format >> {prompt} | " + text
        return result
    
    def one_token(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            One token prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
        result = f"<{se_order}> : " + text
        return result
    
    def no_prompt(self, text:str, se_order:str="aos") -> str:
        """
        ### DESC
            No prompt.
        ### PARAMS
        * text: Text.
        * se_order: Sentiment element order, denoted by a (aspect), c (category), o (opinion), and s (sentiment).
        ### RETURN
        * Prompted text.
        """
        return text
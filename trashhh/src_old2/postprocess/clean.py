from typing import List
class Cleaner:
    """
    Responsible to clean the output from the generative ABSA model.
    """
    def one(self, out:str, remove:List[str]=["</s>","<pad>"]) -> str:
        """
        ### DESC
            Method to clean one instance of decomposed answer.
        ### PARAMS
        * out: Decomposed answer.
        * remove: Phrase/word/token that needs to be removed.
        ### RETURN
        * Clean answer.
        """
        result = out
        for tok in remove:
            result = result.replace(tok, '')
        result = result.strip()
        return result

    def many(self, outputs:List[str], remove:List[str]=["</s>","<pad>"]) -> List[str]:
        """
        ### DESC
            Method to clean many instance of decomposed answer.
        ### PARAMS
        * outputs: List of decomposed answer.
        * remove: Phrase/word/token that needs to be removed.
        ### RETURN
        * List of clean answer.
        """
        return [self.one(out=out, remove=remove) for out in outputs]
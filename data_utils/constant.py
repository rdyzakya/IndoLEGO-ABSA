SENTTAG2WORD = {"POS": "positive", "NEG": "negative", "NEU": "neutral", "MIX" : "mixed"}
SPECIAL_CHAR = {"[": "\[", "]": "\]", ".": "\.", "\\": "\\", "{": "\{", "}": "\}", "^": "\^", "$": "\$", "*": "\*", "+": "\+", "?": "\?", "|": "\|", "(": "\(", ")": "\)"}
PATTERN_TOKEN = {"aspect" : "<A>", "opinion" : "<O>", "sentiment" : "<S>", "category" : "<C>"}
SENTIMENT_ELEMENT = {'a' : "aspect", 'o' : "opinion", 's' : "sentiment", 'c' : "category"}
# NOTE FOR SPECIAL_CHAR:
# // symbol is the same, because the code use replace method with regex parameter set to True
# result example: noodle/spaghetti --> noodle//spaghetti

SEP = "####"

FORMAT_PATTERN_MASK = "PATTERN"
CATEGORY_MASK = "CATEGORY"
IMPUTATION_FIELD_MASK = "IMPUTATION_FIELD"

NO_TARGET = "NONE"

IMPLICIT_ASPECT = "NULL"

CONSTANT_VOCAB = list(SENTTAG2WORD.keys()) + list(SENTTAG2WORD.values()) \
        + list(SPECIAL_CHAR.keys()) + list(SPECIAL_CHAR.values()) \
        + list(PATTERN_TOKEN.keys()) + list(PATTERN_TOKEN.values()) \
        + list(SENTIMENT_ELEMENT.keys()) + list(SENTIMENT_ELEMENT.values()) \
        + [IMPLICIT_ASPECT,NO_TARGET]
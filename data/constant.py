SENTTAG2WORD = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
SPECIAL_CHAR = {"[": "\[", "]": "\]", ".": "\.", "\\": "\\", "{": "\{", "}": "\}", "^": "\^", "$": "\$", "*": "\*", "+": "\+", "?": "\?", "|": "\|", "(": "\(", ")": "\)"}
PATTERN_TOKEN = {"aspect" : "<A>", "opinion" : "<O>", "sentiment" : "<S>", "category" : "<C>"}
SENTIMENT_ELEMENT = {'a' : "aspect", 'o' : "opinion", 's' : "sentiment", 'c' : "category"}
# NOTE FOR SPECIAL_CHAR:
# // symbol is the same, because the code use replace method with regex parameter set to True
# result example: noodle/spaghetti --> noodle//spaghetti

SEP = "####"
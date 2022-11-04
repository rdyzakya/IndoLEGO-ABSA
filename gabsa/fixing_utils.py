import nltk

sentiment_word_list = ['positive', 'negative', 'neutral']
options = ['editdistance', 'remove', 'cut']

def is_term_in_sentence(term,sent):
    splitted_term = term.split()
    splitted_sent = sent.split()
    for j in range(0,len(splitted_sent)-len(splitted_term)+1):
        if splitted_term == splitted_sent[j:j+len(splitted_term)]:
            return True
    return False

def recover_terms_with_editdistance(original_term, sent): # edited
    """
    original_term : str (original term)
    sent : str
    """
    sent = sent.split()
    words = original_term.split(' ')
    terms = {}
    for i in range(len(sent)-len(words)+1):
        # word window
        window = sent[i:i+len(words)]
        new_term = ' '.join(window) # .lower()
        levenshtein_distance = nltk.edit_distance(original_term.lower(), new_term.lower())
        terms[new_term] = levenshtein_distance
    try:
        smallest_term = min(terms, key=terms.get)
    except Exception as e:
        print("Sents:",sent)
        print("Words:",words)
        raise e
    return smallest_term

def recover_term_by_cut(original_term, sent):
    words = original_term.split(' ')
    len_word = len(words)
    while len_word > 0:
        words_combination = [words[:len_word],words[-len_word:]]
        for word in words_combination:
            new_term = ' '.join(word) # .lower()
            if is_term_in_sentence(new_term,sent):
                return new_term
        len_word -= 1
    return original_term

def fix_preds(all_tups, sents, option='editdistance'):

    if option not in options:
        raise ValueError('Invalid option, please choose from {}'.format(options))

    all_new_tups = []

    for i, tups in enumerate(all_tups):
        new_tups = []
        if tups == []:
            all_new_tups.append(tups)
        else:
            new_tups = []
            for tup in tups:
                is_remove = False
                new_tup = {}
                for k,v in tup.items():
                    if k == "sentiment":
                        if v not in sentiment_word_list:
                            continue
                    else:                       
                        if not is_term_in_sentence(v,sents[i]):
                            if option == "remove":
                                is_remove = True
                                break
                            elif option == "editdistance":
                                v = recover_terms_with_editdistance(v,sents[i])
                            elif option == "cut":
                                v = recover_term_by_cut(v,sents[i])
                    new_tup[k] = v
                if not is_remove:
                    new_tups.append(new_tup)
            all_new_tups.append(new_tups)
    
    return all_new_tups
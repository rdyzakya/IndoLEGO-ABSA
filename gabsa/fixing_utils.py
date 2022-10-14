import nltk

sentiment_word_list = ['positive', 'negative', 'neutral']
options = ['editdistance', 'remove', 'cut']

# From Ishak Zhang
def recover_terms_with_editdistance(original_term, sent): # edited
    """
    original_term : str (original term)
    sent : list of str (sentence) | example: ['I', 'am', 'a', 'student']
    """
    words = original_term.split(' ')
    terms = {}
    for i in range(len(sent)-len(words)+1):
        # word window
        window = sent[i:i+len(words)]
        new_term = ' '.join(window).lower()
        levenshtein_distance = nltk.edit_distance(original_term.lower(), new_term)
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
    joined_sent = ' '.join(sent).lower()
    len_word = len(words)
    while len_word > 0:
        words_combination = [words[:len_word],words[-len_word:]]
        for word in words_combination:
            new_term = ' '.join(word).lower()
            if new_term in joined_sent:
                return new_term
        len_word -= 1
    return original_term

def fix_preds_aste(all_pairs, sents, option='editdistance'):

    if option not in options:
        raise ValueError('Invalid option, please choose from {}'.format(options))

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #two formats have different orders (old)
                #target, marker, sentiment (new)
                try:
                    target, marker, sentiment = pair
                except Exception as e:
                    print("Pair:",pair)
                    raise e
                if sentiment not in sentiment_word_list:
                    continue
                    # raise ValueError(f"Polarity {sentiment} is invalid | Target : {target} | Marker : {marker}")
                
                joined_sent = ' '.join(sents[i])

                # remove option if at is not in the sentence
                if target.lower() not in joined_sent.lower() and option == 'remove':
                    continue

                # AT not in the original sentence
                if target.lower() not in  joined_sent.lower():
                    new_target = recover_terms_with_editdistance(target, sents[i]) if option == 'editdistance' else recover_term_by_cut(target, sents[i])
                else:
                    new_target = target
                
                # OT not in the original sentence
                markers = marker.split(', ')
                new_markers_list = []
                for m in markers:
                    if m.lower() not in joined_sent.lower():
                        if option == 'remove':
                            continue
                        elif option == 'editdistance':
                            new_markers_list.append(recover_terms_with_editdistance(m, sents[i]))
                        else:
                            new_markers_list.append(recover_term_by_cut(m, sents[i]))
                    else:
                        new_markers_list.append(m)
                new_marker = ', '.join(new_markers_list)
                # No valid opinion term found
                if new_marker == '':
                    continue
                new_pairs.append((new_target,new_marker,sentiment))
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs
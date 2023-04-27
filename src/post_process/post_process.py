from nltk.metrics.distance import edit_distance

def normalize_phrase(phrase:str, text:str, threshold:float=0.5) -> str:
    """
    ### DESC
    Normalizes a phrase so that it appears in a text by removing any extra spaces and ensuring that
    each word in the phrase is separated by a single space. Uses Levenshtein distance with a windowing
    method to handle typos and spelling mistakes.

    ### PARAMS
    * phrase: The phrase to be normalized.
    * text: The text in which the phrase should appear.
    * threshold: The threshold to determine if a closest prase can replace the original phrase (from 0 to 1.0).

    Returns:
    * The normalized version of the phrase, or None if no match is found.
    """
    # Remove any extra spaces from the phrase
    phrase = " ".join(phrase.split())

    # Split the phrase into individual words
    words = phrase.split(" ")

    # Create a list to hold the normalized words
    normalized_words = []

    # Normalize each word in the phrase by adding it to the normalized_words list
    for word in words:
        normalized_word = word.strip().lower()
        normalized_words.append(normalized_word)

    # Join the normalized words into a single string with a single space between each word
    normalized_phrase = " ".join(normalized_words)

    # Compute the Levenshtein distance between the normalized phrase and each window of words in the text
    closest_match = None
    closest_distance = float('inf')
    for i in range(len(text.split(" ")) - len(normalized_words) + 1):
        window = " ".join(text.split(" ")[i:i+len(normalized_words)])
        distance = edit_distance(normalized_phrase, window.lower())
        if distance < closest_distance:
            closest_match = window
            closest_distance = distance

    # If the closest match is within a threshold, consider it a match
    if closest_distance <= len(normalized_phrase) * threshold:
        return closest_match
    else:
        return None


if __name__ == "__main__":
    text = "This is a sample text string."
    phrase = "SAAMPLE  txt"

    normalized_phrase = normalize_phrase(phrase, text)

    if normalized_phrase:
        print(f"The normalized phrase '{normalized_phrase}' appears in the text.")
    else:
        print(f"The normalized phrase '{phrase}' does not appear in the text.")

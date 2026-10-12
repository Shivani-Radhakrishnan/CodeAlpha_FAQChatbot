from botbuddy.text import content_tokens, normalize, stemmed_text, tokenize


def test_normalize_expands_contractions_and_curly_quotes():
    assert normalize("Where’s my order?") == "where is my order"
    assert normalize("I can't log in, I'm stuck") == "i can not log in i am stuck"
    assert normalize("It hasn't arrived") == "it has not arrived"


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Café — open 24/7!!") == "cafe open 24 7"


def test_tokenize_empty():
    assert tokenize("   ?! ") == []


def test_content_tokens_drop_stopwords_but_keep_intent_words():
    assert content_tokens("How do I reset my password?") == ["reset", "password"]
    assert content_tokens("help") == ["help"]
    # A sentence made only of stop words keeps its tokens rather than vanishing.
    assert content_tokens("who are you") == ["who", "are", "you"]


def test_stemming_merges_inflections():
    assert stemmed_text("returning returns") == stemmed_text("return return")

from observatoire.utils import tokenize_keywords


def test_tokenize_keywords_splits_on_spaces_and_commas():
    assert tokenize_keywords("Data Science, Python   engineer") == ["data", "science", "python", "engineer"]

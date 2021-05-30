from typing import List, Tuple


def ngram_contained(sent: List[str], bi: List[str], n=2) -> bool:
    """ Check if an n-gram is contained inside a sentence, given as a list ot tokens

    :param sent: The sentence, given as a List of strings (tokens)
    :param bi: The n-gram to check against, a tuple/list of strings (tokens) with size `n`
    :param n: The size of the n-grams. For example `n=2` expects a bi-gram and `n=1` expects a uni-gram (single token).
    :return: True if the n-gram is contained in the sentence, False otherwise.
    """
    for i in range(n - 1, len(sent)):
        contained = all(bi[j] == sent[i - (n - 1 - j)] for j in range(n))
        if contained:
            return True
    return False


def words_to_ngrams(words: List[str]) -> List[Tuple[str]]:
    """ Convert a list of words to uni-grams

    :param words: The list of words to convert
    :return: A list of the same size, containing single-element tuples of uni-grams.
    """
    return [(w,) for w in words]


def ngrams_to_tuples(ngrams: List[str]) -> List[Tuple]:
    """
    Convert a list of strings to n-grams. Does not check that all n-grams have the same length.
    :param ngrams: The list of n-grams, given as space-delimited strings.
    :return: A list of tuples, each representing an n-gram.
    """
    return [tuple(w.split(' ')) for w in ngrams]

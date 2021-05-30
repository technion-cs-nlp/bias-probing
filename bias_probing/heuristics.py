# These codes are adapted from the codes for
# Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference by
# Tom McCoy, Ellie Pavlick, Tal Linzen, ACL 2019
import re
from typing import Tuple, List, Union

from nltk import word_tokenize

_IGNORED_TOKENS = [".", "?", "!", "-"]


def sentence_to_words(sent: Union[str, List[str]], ignored: List[str] = None, lowercase=True):
    if ignored is None:
        ignored = _IGNORED_TOKENS
    if isinstance(sent, str):
        sent = word_tokenize(sent)

    assert isinstance(sent, list)
    regex = re.compile('[' + "".join(ignored).replace('.', r'\.').replace('?', r'\?').replace('-', r'\-') + ']')
    if lowercase:
        return [regex.sub('', word.lower()) for word in sent if word not in ignored]
    else:
        return [regex.sub('', word) for word in sent if word not in ignored]


def _prem_hypothesis_to_words(premise: str, hypothesis: str, lowercase=True):
    prem_words = sentence_to_words(premise, lowercase=lowercase)
    hyp_words = sentence_to_words(hypothesis, lowercase=lowercase)
    return prem_words, hyp_words


def have_lexical_overlap(premise: str, hypothesis: str, get_hans_new_features=False, lowercase=True) -> Tuple[bool, float]:
    r"""Check if a given premise and hypothesis lexically overlap.

    :param premise: The premise
    :param hypothesis: The hypothesis
    :param get_hans_new_features: If True, the returned overlap percentage is calculated w.r.t. the hypothesis.
    Otherwise, it is calculated w.r.t. the premise.

    :return:
        all_in (bool): True if all the words in the hypothesis are also in the premise, False otherwise.
        overlap_percent (int): The percentage of overlapping words (types) in the hypothesis the are also in
        the premise.
    """
    prem_words, hyp_words = _prem_hypothesis_to_words(premise, hypothesis, lowercase=lowercase)
    all_in = True

    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break

    num_overlapping = len(list(set(hyp_words) & set(prem_words)))
    if get_hans_new_features:
        overlap_percent = num_overlapping / len(set(hyp_words))
    else:
        overlap_percent = num_overlapping / len(set(prem_words)) if len(set(prem_words)) > 0 else 0

    return all_in, overlap_percent


def is_subsequence(premise: str, hypothesis: str, lowercase=True) -> bool:
    r"""Check if a given premise contains a given hypothesis as a subsequence.

    :param premise: The premise
    :param hypothesis: The hypothesis

    :return:
        is_subsequence (bool): True if the hypothesis is a sequence of the premise, False otherwise.
    """
    prem_words, hyp_words = _prem_hypothesis_to_words(premise, hypothesis, lowercase=lowercase)

    prem_filtered = f'|{"|".join(prem_words)}|'
    hyp_filtered = f'|{"|".join(hyp_words)}|'

    return hyp_filtered in prem_filtered


# noinspection DuplicatedCode
def _parse_phrase_list(parse, phrases):
    if parse == '':
        return phrases

    phrase_list = phrases

    # Normalize parse for easy processing
    parse = parse.replace('( ', '(').replace(' )', ')').replace('(', '( ').replace(')', ' )')
    words = parse.split()
    this_phrase = []
    next_level_parse = []
    for index, word in enumerate(words):
        if word == "(":
            next_level_parse += this_phrase
            this_phrase = ["("]
        elif word == ")" and len(this_phrase) > 0 and this_phrase[0] == "(":
            phrase_list.append(" ".join(this_phrase[1:]))
            next_level_parse += this_phrase[1:]
            this_phrase = []
        elif word == ")":
            next_level_parse += this_phrase
            next_level_parse.append(")")
            this_phrase = []
        else:
            this_phrase.append(word)
    return _parse_phrase_list(" ".join(next_level_parse), phrase_list)


def is_constituent(premise, hypothesis, premise_parse):
    """Check if a given premise's parse tree contains a given hypothesis as a subtree.

    Parameters:
        premise (str): The premise
        hypothesis (str): The hypothesis
        parse (str): A string representing the premise's parse tree, in binary-branching format
        (See MNLI for information).
    """
    ignored = _IGNORED_TOKENS
    parse_new = [word.lower() for word in premise_parse.split() if word not in ignored]

    all_phrases = _parse_phrase_list(" ".join(parse_new), [])

    # prem_words = sentence_to_words(premise, ignored, lowercase=True)
    hyp_words = sentence_to_words(hypothesis, ignored, lowercase=True)

    hyp_filtered = ' '.join(hyp_words)
    return hyp_filtered in all_phrases

from collections import defaultdict
from math import log
from typing import Callable

from tqdm import tqdm


def get_ngram_label_counts(dataset, preprocess_fn: Callable, verbose=True, n=1):
    label_counts = defaultdict(int)
    word_counts = defaultdict(lambda: 1)
    word_label_counts = defaultdict(lambda: 1)
    for sample in tqdm(dataset, desc='Calculating...', disable=not verbose):
        tokenized, label = preprocess_fn(sample)
        words = [w for w in tokenized if w not in [',', '.', '!', '?']]
        label_counts[label] += 1
        for i in range(n, len(words)):
            ngram = tuple(words[i - n:i])
            # print(ngram)
            word_label_counts[(ngram, label)] += 1
            word_counts[ngram] += 1

    print(f'Labels: {label_counts.keys()}')
    return word_counts, label_counts, word_label_counts


def get_ngram_label_lmi(dataset, preprocess_fn: Callable, verbose=True, n=1):
    word_counts, label_counts, word_label_counts = get_ngram_label_counts(dataset, preprocess_fn, verbose, n)
    dataset_size = len(dataset)

    word_lmi_dict = defaultdict(dict)
    for word, word_count in word_counts.items():
        for label, label_count in label_counts.items():
            wlc = word_label_counts[(word, label)]
            lmi = (wlc / dataset_size) * (log(wlc) + log(dataset_size) - log(word_count) - log(label_count))
            word_lmi_dict[word][label] = lmi

    return word_lmi_dict


def highest_lmi_words(label, word_lmi_dict, n=10):
    sorted_lmis = []
    for word, label_lmis in word_lmi_dict.items():
        sorted_lmis.append((word, label_lmis[label]))

    return sorted(sorted_lmis, key=lambda tup: tup[1], reverse=True)[:n]

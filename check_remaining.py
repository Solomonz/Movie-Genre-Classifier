#! /usr/bin/env python3

from itertools import chain, combinations
from json import load


with open('all_recorded_accuracies', 'r') as accuracies_file:
    processed = set(map(lambda line: line.strip().split(':')[0], accuracies_file))

with open('processed/genres.json', 'r') as genres_file:
    genres = sorted(load(genres_file).keys())


def p(s, k):
    return chain.from_iterable(combinations(s, r) for r in range(1, k + 1))

def pairs_max_size(k):
    subsets = list(p(genres, k))
    return filter(lambda c: len(set(c[0]) & set(c[1])) == 0, combinations(subsets, 2))

print(len(set(map(str, pairs_max_size(3))) - processed))
print(len(processed))


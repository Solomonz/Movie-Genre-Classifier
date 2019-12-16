from subprocess import run, PIPE
from random import randint
from json import load
from itertools import chain, combinations



with open('processed/genres.json', 'r') as genres_file:
    genres = sorted(load(genres_file).keys())


def powerset_max_size(k):
    return chain.from_iterable(combinations(genres, r) for r in range(1, k + 1))


def pairs_max_size(k):
    subsets = list(powerset_max_size(k))
    return filter(lambda c: len(set(c[0]) & set(c[1])) == 0, combinations(subsets, 2))


all_pairs_max_size = list(pairs_max_size(3))


while True:
    n = randint(0, len(all_pairs_max_size) - 1)
    output = run(['bash', '-c', 'cat all_recorded_accuracies | cut -f 1 -d ":" | sort -u'], stdout=PIPE, stderr=PIPE).stdout.decode(encoding='UTF-8')
    e = all_pairs_max_size[n]
    if output.find(str(e)) == -1:
        print('doing {}'.format(e))
        run(['bash', '-c', 'python3 binary_classifier.py {} 3'.format(n + 1)], stdout=PIPE, stderr=PIPE)
    else:
        print('skipping {}'.format(e))


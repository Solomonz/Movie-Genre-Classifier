from collections import Counter
from itertools import combinations
from json import load

def main():
    single_counts = Counter()
    double_counts = Counter()
    #triple_counts = Counter()

    all_labels = []

    with open('processed/genres.json', 'r') as genres_file:
        genres = load(genres_file)

    id_to_genre = {}
    for genre, gid in genres.items():
        id_to_genre[gid] = genre

    with open('processed/labels.txt', 'r') as labels_file:
        for line in labels_file:
            all_labels.append(list(map(id_to_genre.get, line.strip().split(','))))

    for labels in all_labels:
        single_counts.update(labels)
        double_counts.update(map(tuple, map(sorted, combinations(labels, 2))))
        #triple_counts.update(map(tuple, map(sorted, combinations(labels, 3))))


    print(single_counts)
    print()
    print(double_counts)
    #print()
    #print(triple_counts)

if __name__ == "__main__":
    main()

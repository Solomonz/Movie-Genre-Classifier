import numpy as np
from json import load, loads
from get_data import get_binary_data


with open('processed/genres.json', 'r') as genres_file:
    genres = sorted(load(genres_file).keys())


with open('all_recorded_accuracies', 'r') as output_file:
    data = set(map(lambda line: tuple(line.strip().split(':')), output_file))


data = list(map(lambda datum: (loads(datum[0].replace('(', '[').replace(')', ']').replace(',]', ']').replace("'", '"')), float(datum[1])), filter(lambda l: len(l) == 2, data)))


def class_breakdown_is_significant(equiv_classes):
    ec1, ec2 = tuple(map(set, equiv_classes))
    _, train_labels, _, test_labels, _ = get_binary_data(0, ec1, ec2, shuffle=False)

    total_breakdown = np.sum(train_labels, axis=0) + np.sum(test_labels, axis=0)
    return min(total_breakdown) / max(total_breakdown) > 0.33

data = sorted(data, key=lambda datum: datum[1], reverse=True)

def find_first_acceptable(filter_func):
    for datum in data:
        if filter_func(datum):
            if class_breakdown_is_significant(tuple(map(tuple, datum[0]))):
                return datum
            else:
                data.remove(datum)


    return None

def find_last_acceptable(filter_func):
    for datum in reversed(data):
        if filter_func(datum):
            if class_breakdown_is_significant(tuple(map(tuple, datum[0]))):
                return datum
            else:
                data.remove(datum)

    return None


print('best 1-to-1: {}'.format(find_first_acceptable(lambda datum: len(datum[0][0]) == 1 and len(datum[0][1]) == 1)))
print('worst 1-to-1: {}'.format(find_last_acceptable(lambda datum: len(datum[0][0]) == 1 and len(datum[0][1]) == 1)))
print('best 2-to-2: {}'.format(find_first_acceptable(lambda datum: len(datum[0][0]) == 2 and len(datum[0][1]) == 2)))
print('worst 2-to-2: {}'.format(find_last_acceptable(lambda datum: len(datum[0][0]) == 2 and len(datum[0][1]) == 2)))
print('best 3-to-3: {}'.format(find_first_acceptable(lambda datum: len(datum[0][0]) == 3 and len(datum[0][1]) == 3)))
print('worst 3-to-3: {}'.format(find_last_acceptable(lambda datum: len(datum[0][0]) == 3 and len(datum[0][1]) == 3)))
print('highest accuracy: {}'.format(find_first_acceptable(lambda _: True)))
print('lowest accuracy: {}'.format(find_last_acceptable(lambda _: True)))

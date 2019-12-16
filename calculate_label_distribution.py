from collections import Counter
from csv import DictWriter
from xlsxwriter import Workbook
from itertools import combinations
from json import load

def main():
    single_counts = Counter()
    double_counts = Counter()

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


    print(single_counts)
    print()
    print(double_counts)
    
    workbook = Workbook('distribution.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
    bold_header = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#a0a0a0'})
    bold_header_sideways = workbook.add_format({'bold': True, 'rotation': 90, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#a0a0a0'})


    row = 0
    for genre2 in sorted(single_counts.keys()):
        worksheet.write(row, 0, genre2, bold_header)
        col = 1
        for genre1 in filter(lambda g1: g1 < genre2, sorted(single_counts.keys())):
            co_occurrances = double_counts[(genre1, genre2)]
            ratio = (256 * co_occurrances) // (single_counts[genre1] + single_counts[genre2] - co_occurrances)
            f = workbook.add_format({'bg_color': "#FF{0:02x}{0:02x}".format(255 - (ratio * 3)), 'align': 'center', 'valign': 'vcenter'})
            worksheet.write(row, col, co_occurrances, f)
            col += 1

        worksheet.write(row, row, single_counts[genre2], bold)

        row += 1

    for col, genre1 in enumerate([''] + sorted(single_counts.keys())):
        worksheet.write(len(single_counts.keys()), col, genre1, bold_header_sideways)

    workbook.close()

    with open('distribution.csv', 'w') as dist_file:
        writer = DictWriter(dist_file, fieldnames=['genre'] + sorted(single_counts.keys()))
        writer.writeheader()
        
        for genre2 in sorted(single_counts.keys()):
            row = {'genre': genre2, genre2: single_counts[genre2]}
            for genre1 in filter(lambda g1: g1 < genre2, single_counts.keys()):
                row[genre1] = double_counts[(genre1, genre2)]

            writer.writerow(row)


if __name__ == "__main__":
    main()

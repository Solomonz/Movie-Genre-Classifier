numsDict = {}
genres = []

ind = 0

with open('genre_to_id.txt', 'r') as file:
    for line in file:
        num, genre = line.strip().split(":")
        numsDict[num] = str(ind)
        ind += 1
        genres.append(genre)


with open('updated_genres.txt', 'w') as file:
    for n in numsDict:
        file.write("{}:{}\n".format(n, numsDict[n]))

labels=[]
with open('labels.txt', 'r') as file:
    for line in file:
        labels.append(line.strip().split(","))

for l in range(len(labels)):
    labels[l] = [str(numsDict[n.strip()]) for n in labels[l]]

# print(labels)
with open('labels_converted.txt', 'w') as file:
    for l in labels:
        file.write(" ".join(l))
        file.write('\n')

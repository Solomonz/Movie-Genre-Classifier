lengths = dict()
with open('tokens_2.txt', 'r') as file:
    for line in file:
        curr = len(line.split(','))
        if curr in lengths:
            lengths[curr]+=1
        else:
            lengths[curr]=1

print(sorted(lengths.items(), key=lambda kv:(kv[0],kv[1]), reverse=True))
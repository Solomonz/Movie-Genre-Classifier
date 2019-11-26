import csv, re, nltk
from tqdm import tqdm

def string_to_list(string):
    return re.findall(r"'id': (\d+), 'name': '(\w+)'",string)
def sort(dictionary, rev=False):
    return sorted(dictionary.items(), key=lambda kv:(kv[1], kv[0]), reverse=rev)

def parse_row(row, genre_dict):
    ovv = row[9].split(" ")
    if len(ovv)>5:
        genres = set()
        gid = string_to_list(row[3])
        # print(gid)
        if gid is not None and len(gid)>0:
            for g in gid:
                if g[0] not in genre_dict:
                    genre_dict[g[0]] = g[1]
                genres.add(str(g[0]))
            #imdb, 
            return row[5], row[6], genres, row[9] 
    return None

filename = '../the-movies-dataset/movies_metadata.csv'
# reader = csv_reader()

data = []
indices = [5,6,3,9] 

f=0
s=0

id_to_genre = dict()
synopses = []
labels = []
vocab = dict()


#Get genre and synposis from CSV
with open(filename,'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    # with open(r'simple_data.txt', 'w', encoding='utf-8') as file:
    for row in reader:
        parsed = parse_row(row, id_to_genre)
        if parsed is not None:
            labels.append(parsed[2])
            text = parsed[3].lower()
            synopses.append(text)

# print(id_to_genre)

#Count lemmatize verbs and nouns:
lemmas = dict()
lemmer = nltk.stem.WordNetLemmatizer()

for s in tqdm(synopses):
    pos = nltk.pos_tag(nltk.word_tokenize(s))
    
    for pair in pos:
        #lemmatized version
        token = pair[0]
        part = pair[1].lower()[0]
        if part =='n' or part =='v':
            token = lemmer.lemmatize(token, part)
        
        if token in lemmas:
            lemmas[token]+=1
        else:
            lemmas[token]=1

# print(len(lemmas))

tokens = []
vocab_key = dict()
ind = 0

for s in tqdm(synopses):
    sent = []
    pos = nltk.pos_tag(nltk.word_tokenize(s))

    for pair in pos:
        #lemmatized version
        token = pair[0]
        part = pair[1].lower()[0]
        if part =='n' or part =='v':
            token = lemmer.lemmatize(token, part)
        
        if lemmas[token]<100:
            token = "unk"+part
        
        if token not in vocab_key:
            vocab_key[token] = ind
            ind+=1                    
        sent.append(str(vocab_key[token]))
    
    tokens.append(sent)


print(len(tokens))
print(len(synopses))
print(len(labels))

with open('tokens.txt', 'w', encoding='utf-8') as file:
    for t in tokens:
        file.write(",".join(t))
        file.write('\n')


with open('synopses.txt', 'w',encoding='utf-8') as file:
    for s in synopses:
        file.write(s)
        file.write('\n')

with open('labels.txt', 'w', encoding='utf-8') as file:
    for l in labels:
        file.write(",".join(l))
        file.write('\n')

with open('vocab.txt', 'w',encoding='utf-8') as file:
    for v in vocab_key:
        file.write(str(v))
        file.write(" ")
        file.write(str(vocab_key[v]))
        file.write('\n')

    
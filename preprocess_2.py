import csv, re, nltk
from collections import defaultdict
from tqdm import tqdm
import numpy 
import string
import json
# try:
#     nltk.pos_tag(nltk.word_tokenize("doing"))
#     nltk.stem.WordNetLemmatizer().lemmatize("doing", "do")
# except LookupError:
#     nltk.download()

#Add Documentary?

stemmer = nltk.stem.snowball.SnowballStemmer("english")



resplit = re.compile("(?<! St)(?<! temp)(?<! tel)(?<! misc)(?<! col)(?<! sgt)(?<! dr)(?<! St)(?<! prof)(?<! rev)(?<! hon)(?<! esq)(?<! approx)(?<! apt)(?<! appt)(?<! mt)(?<! Rd)(?<! ave)(?<! etc)(?<! jr)(?<! Sr)(?<! Mr)(?<! mrs)(?<! ms)(?<![ \.]\w)[;!?.]", re.IGNORECASE)

useless = {"Animation", "Family", "Music", "Foreign", "TV Movie"}
genre_dict = dict()

#Splits a sentence into list of stemmed strings (string tokens) 
#TODO: REMOVE ALL OTHER PUNCTUATION!
def splitter(tosplit, resplit = resplit, stemmer=stemmer):
    splitted = resplit.split(tosplit)
    splitted = [s.translate(str.maketrans('', '', string.punctuation)) for s in splitted] 
    splitted = [[stemmer.stem(x) for x in s.split()] for s in splitted]
    return [s for s in splitted if s != []]

#Take the ugly genre string from csv and convert to list of genres 
def string_to_list(string):
    return re.findall(r"'id': \d+, 'name': '([\w\s]+)'",string)

#Sorts dictionary if you'd like
def sort(dictionary, rev=False):
    return sorted(dictionary.items(), key=lambda kv:(kv[1], kv[0]), reverse=rev)

#Returning set of genres (may be empty) and list of sentences (split/stemmed)
def parse_row(row, genre_dict=genre_dict, uesless=useless):
    ovv = row[9].split(" ")
    if len(ovv)>5:
        genres = set()
        gid = string_to_list(row[3])
        # print(gid)
        if gid is not None and len(gid)>0:
            for g in gid:
                if g[0] not in useless:
                    if g[0] not in genre_dict:
                        genre_dict[g[0]] = str(len(genre_dict))
                    genres.add(str(genre_dict[g[0]]))
            return genres, splitter(row[9])
    return None

# reader = csv_reader()

data = []
f=0
s=0
synopses = []
labels = []

filename = 'data/the-movies-dataset/movies_metadata.csv'
#Get genre and synposis from CSV
# READ FROM FILE--> SHOULD CHANGE
with open(filename,'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in tqdm(reader):
        parsed = parse_row(row, genre_dict)
        if parsed is not None and len(parsed[0]) >0 and len(parsed[1])>0:
            labels.append(parsed[0])
            synopses.append(parsed[1])

if len(labels) != len(synopses):
    raise Exception(f'Preprocessing error: label length {len(labels)} did not equal synopses lengths {len(synopses)}')


vocab = dict()
#Vocab builder
for syn in synopses:
    for sent in syn:
        for word in sent:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1



threshold = 25
#Removes uncommon words
vocab = {k:v  for k,v in vocab.items() if v > threshold}
#Indices 0 (PAD) and 1 (SENTENCE/BREAK) are reserved 

vocab["*pad*"] =0 

#Else/Unknown
vocab["*UNK*"]=1

# nltk.tag.pos_tag(['going'])
vocab["*UNK_N*"]=2
vocab["*UNK_V*"]=3
vocab["*UNK_A*"]=4
# Sentences
vocab["*BREAK*"]=5


ind = 6
for v in vocab:
    vocab[v] = str(ind)
    ind +=1

#TODO: Instead of saving to txt, save as JSON (avoid line-overflow)
#TODO: Decide whether each point is sentence or movie (currently split at sentences)
#TODO: If split at movie, add "*BREAK*" at end of each sentence
with open('tokens_2.txt', 'w') as tokens_file, open('labels_2.txt', 'w') as labels_file:
    for i in tqdm(range(len(synopses))):
        for sent in synopses[i]:
            if len(sent)>=5:
                towrite = []
                for word in sent:
                    #TODO: SEMANTIC UNK
                    # nltk.tag.pos_tag(['going'])--> check if first letter is N, V, A (else use regular unk)

                    if word in vocab:
                        towrite.append(vocab[word])
                    else:
                        towrite.append('1')
                tokens_file.write(",".join(towrite))
                tokens_file.write('\n')
                
                labels_file.write(",".join(labels[i]))
                labels_file.write('\n')

#TODO: Fix the json

with open('genres_2.txt', 'w') as file:
    file.write(json.dumps(genre_dict))

with open('vocab_2.txt', 'w') as file:
    file.write(json.dumps(vocab))

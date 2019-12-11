import re, nltk
from collections import defaultdict
from csv import DictReader
from tqdm import tqdm
import numpy 
import string
from json import dump, loads
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
def splitter(tosplit, resplit=resplit, stemmer=stemmer):
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
def parse_row(row):
    ovv = row['overview']
    if len(ovv) > 5:
        #gid = string_to_list(row['genres'])
        labels = set()
        genres = set(map(lambda inner_genre_dict: inner_genre_dict['name'], loads(row['genres'].replace("'", '"')))) - useless
        if len(genres) > 0:
            for genre in genres:
                if genre not in genre_dict:
                    genre_dict[genre] = str(len(genre_dict))
                labels.add(genre_dict[genre])
            return labels, splitter(row['overview'])
    return None, None


data = []
f = 0
s = 0
synopses = []
labels = []

filename = 'data/the-movies-dataset/movies_metadata.csv'
#Get genre and synposis from CSV
# READ FROM FILE--> SHOULD CHANGE
with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = DictReader(csvfile)
    for row in tqdm(reader):
        genres, tokens = parse_row(row)
        if genres is not None and len(tokens) > 0:
            labels.append(genres)
            synopses.append(tokens)

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

vocab["*pad*"] = 0 

#Else/Unknown
vocab["*UNK*"] = 0

# nltk.tag.pos_tag(['going'])
vocab["*UNK_N*"] = 0
vocab["*UNK_V*"] = 0
vocab["*UNK_A*"] = 0
# Sentences
vocab["*BREAK*"] = 0


ind = 0
for v in vocab:
    vocab[v] = str(ind)
    ind +=1

#TODO: Instead of saving to txt, save as JSON (avoid line-overflow)
#TODO: Decide whether each point is sentence or movie (currently split at sentences)
#TODO: If split at movie, add "*BREAK*" at end of each sentence
with open('processed/tokens.txt', 'w') as tokens_file, open('processed/labels.txt', 'w') as labels_file:
    for i in tqdm(range(len(synopses))):
        for sent in synopses[i]:
            towrite = []
            if len(sent)>=5:
                for word, tag in nltk.tag.pos_tag(sent):
                    #TODO: SEMANTIC UNK
                    # nltk.tag.pos_tag(['going'])--> check if first letter is N, V, A (else use regular unk)

                    if word in vocab:
                        towrite.append(vocab[word])
                    else:
                        if tag[0] in {'N', 'V', 'A'}:
                            towrite.append(vocab['*UNK_{}*'.format(tag[0])])
                        else:
                            towrite.append(vocab['*UNK*'])
                towrite.append(vocab['*BREAK*'])
        if len(towrite) == 0:
            continue
        tokens_file.write(",".join(towrite))
        tokens_file.write('\n')
        
        labels_file.write(",".join(labels[i]))
        labels_file.write('\n')


with open('processed/genres.json', 'w') as genres_file:
    dump(genre_dict, genres_file, separators=(',', ':'))

with open('processed/vocab.json', 'w') as vocab_file:
    dump(vocab, vocab_file, separators=(',', ':'))

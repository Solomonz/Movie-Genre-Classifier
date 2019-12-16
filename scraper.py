#! /usr/bin/env python3

from csv import DictReader
from bs4 import BeautifulSoup
import requests
from json import dump, load
from tqdm import tqdm
from random import shuffle


def scrape_synopsis(imdb_id):
    s = requests.Session()
    s.trust_env = False
    req = s.get("https://www.imdb.com/title/{}/plotsummary".format(imdb_id))
    soup = BeautifulSoup(req.text, 'html.parser')
    synopsis_h4 = soup.find('h4', string="Synopsis")
    synopsis_li = synopsis_h4.nextSibling.nextSibling.find('li')
    return synopsis_li.getText()


def save_to_file(synopses):
    with open('processed/full_synopses.json', 'r') as synopses_file:
        existing = load(synopses_file)

    existing.update(synopses)

    with open('processed/full_synopses.json', 'w') as synopses_file:
        dump(existing, synopses_file, indent=None, separators=(',', ':'))


def main():
    with open('processed/full_synopses.json', 'r') as synopses_file:
        known_synopses = load(synopses_file)

    with open('data/the-movies-dataset/movies_metadata.csv', 'r') as data_file:
        reader = DictReader(data_file)
        imdb_ids = set()
        for row in reader:
            imdb_ids.add(row['imdb_id'])
                
    imdb_ids.difference_update(known_synopses)
    imdb_ids = list(imdb_ids)
    shuffle(imdb_ids)

    imdbID_to_synopsis = {}

    for imdb_id in tqdm(imdb_ids):
        try:
            imdbID_to_synopsis[imdb_id] = scrape_synopsis(imdb_id)
        except Exception as e:
            import pdb;pdb.set_trace()
            pass

        if len(imdbID_to_synopsis) >= 200:
            save_to_file(imdbID_to_synopsis)
            return


    save_to_file(imdbID_to_synopsis)


if __name__ == "__main__":
    while True:
        main()

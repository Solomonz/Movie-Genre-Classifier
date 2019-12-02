#! /usr/bin/env python3

from bs4 import BeautifulSoup
import requests
from json import dump
from tqdm import tqdm


def scrape_synopsis(imdb_id):
    req = requests.get("https://www.imdb.com/title/{}/plotsummary".format(imdb_id))
    soup = BeautifulSoup(req.text, 'html.parser')
    synopsis_h4 = soup.find('h4', string="Synopsis")
    synopsis_li = synopsis_h4.nextSibling.nextSibling.find('li')
    return synopsis_li.getText()


def main():
    with open('all_simple_data.txt', 'r') as data_file:
        next(data_file)  # skip past the header row
        imdb_ids = set()
        for line in data_file:
            tokens = line.strip().split('|||')
            if len(tokens) >= 3 and tokens[2][:2] == 'tt':
                imdb_ids.add(tokens[2])
                
        #imdb_ids = {line.strip().split('|||')[2] for line in data_file}

    imdbID_to_synopsis = {}

    for imdb_id in tqdm(imdb_ids):
        imdbID_to_synopsis[imdb_id] = scrape_synopsis(imdb_id)

    with open('full_synopses.json', 'w') as synopses_file:
        print('dumping synopses')
        dump(imdbID_to_synopsis, synopses_file, indent=None, separators=(',', ':'))
        print('finished dumping synopses')


if __name__ == "__main__":
    main()

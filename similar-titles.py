import numpy as np
import pandas as pd
import random
import spacy
from annoy import AnnoyIndex

nlp = spacy.load('en_core_web_md')

df = pd.read_csv('data/ViewingActivity.csv')
raw_titles = df['Title'].str.split('[:(_]', regex=True).str[0]
raw_titles = raw_titles.str.strip()
titles_cleaned = raw_titles.drop_duplicates()

titles = {}
for title in titles_cleaned:
	print(f'[INFO] Adding {title} to data')
	sent = nlp(title)
	titles[sent.text] = sent.vector

# print(titles['Stranger Things'])

titles_lookup = AnnoyIndex(300, 'angular')
idx = 0
id_map = {}
corpus = []
for t, v in titles.items():
	titles_lookup.add_item(idx, v)
	id_map[t] = idx
	corpus.append(t)
	idx += 1

titles_lookup.build(10)
print([corpus[i] for i in titles_lookup.get_nns_by_vector(nlp('Kingdom').vector, 10)])
#print([corpus[i] for i in titles_lookup.get_nns_by_vector('titles['Eat Pray Love']', 10)])
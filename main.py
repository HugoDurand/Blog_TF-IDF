import html
import re
from pprint import pprint

import feedparser

# https://www.ranks.nl/stopwords/french
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

feed = feedparser.parse('https://blog.eleven-labs.com/feed.xml')

articles = feed['entries']

print(len(articles))
exit()

def clean_text(text):
    # retire les tags html
    clean_text = re.sub(re.compile('<.*?>'), '', text)
    # ça je sais pas
    clean_text = html.unescape(clean_text).replace(u'\xa0', ' ')
    # retire les caracteres non alphanumerique sauf baqique ponctuation
    clean_text = re.sub(r'([^\w,\.:;!?\'-\(\)]|_)', ' ', clean_text)
    # ça je sais pas
    clean_text = re.sub(r'\[scald=[^]]+\]', '', clean_text)
    # retire les espaces blanc
    clean_text = re.sub(r'\s+', r' ', clean_text).strip()

    return clean_text


frenchCorpus = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il',
                'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos',
                'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son',
                'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j',
                'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants',
                'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez',
                'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez',
                'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient',
                'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants',
                'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons',
                'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions',
                'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient',
                'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent']


# paramètres nécéssaires :
nb_neighbours = 4
vect_data = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                            stop_words=frenchCorpus,
                            token_pattern=r'\b[^\d\W]+\b',
                            min_df=0.003,
                            max_df=0.5,
                            max_features=5000)
near_neigh_data = NearestNeighbors(n_neighbors=nb_neighbours, algorithm='brute', metric='cosine')


# Convert to dataframe
data = []
columns_to_keep = ["link", "description"]
for d in articles:
    try:
        d['description']
    except KeyError:
        d['description'] = ""
    try:
        d['link']
    except KeyError:
        d['link'] = ""
    try:
        data.append([d[col] for col in columns_to_keep])
    except KeyError as missing_col:
        print(missing_col)
data = pd.DataFrame(data=data, columns=columns_to_keep)
data['description'] = data['description'].apply(clean_text)

def chunks(liste, ntotal):
    for i in range(0, len(liste), ntotal):
        yield liste[i:i + ntotal]

# calcul similarité et distance
tfidf_data = vect_data.fit_transform(data["description"])
tfidf = tfidf_data.copy()
matching = data["link"].reset_index().rename(columns={"index": "item"})
nearest_neighbours = near_neigh_data.fit(tfidf)
liste_distance_matrix = []

working_chunk = 1
for chunk in chunks(range(tfidf_data.shape[0]), 1000):
    working_chunk = working_chunk + 1
    liste_distance_matrix.append(np.hstack(nearest_neighbours.kneighbors(tfidf_data[chunk], return_distance=True)))
distance_matrix = np.vstack(liste_distance_matrix)

distance_matrix = pd.DataFrame(distance_matrix, columns=["dist_" + str(i) for i in range(0, nb_neighbours)] +
                                                        ["item_" + str(i) for i in range(0, nb_neighbours)])

for col in ["item_" + str(i) for i in range(0, nb_neighbours)]:
    distance_matrix[col] = distance_matrix[col].astype(int)

for i in range(0, nb_neighbours):
    distance_matrix = distance_matrix.merge(
        matching.rename(columns={"item": "item_" + str(i), "link": "link_" + str(i)}), how="left")

distance_matrix = pd.concat([distance_matrix, data["link"].reset_index().rename(columns={"index": "item"})], axis=1)

for i in reversed(range(1, nb_neighbours)):
    distance_matrix["link_" + str(i)] = np.where(distance_matrix['link'] != distance_matrix['link_0'],
                                                 distance_matrix["link_" + str(i - 1)],
                                                 distance_matrix["link_" + str(i)])
    distance_matrix["dist_" + str(i)] = np.where(distance_matrix['link'] != distance_matrix['link_0'],
                                                 distance_matrix["dist_" + str(i - 1)],
                                                 distance_matrix["dist_" + str(i)])

distance_matrix = distance_matrix[["link"] + ["link_" + str(i) for i in range(1, nb_neighbours)] +
                                  ["dist_" + str(i) for i in range(1, nb_neighbours)]]

print(distance_matrix['link_3'][8])
exit()



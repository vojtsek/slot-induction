from induct import Embedding
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import json
import spacy
import pickle
import sys
import glob

frames_vocab = {}
pos_vocab = {}
word_vocab = {}
stopwords = set(stopwords.words('english'))

embeddings = Embedding('~/hudecek/data/elmo/elmo_camrest_embs.txt')
clustering = AgglomerativeClustering(n_clusters=10, linkage='average', affinity='cosine')
embs = np.array([item for _, item in embeddings.items()])
clustering = clustering.fit(embs)
word_dict = {}
for idx, it in enumerate(embeddings.items()):
    word_dict[it[0]] = clustering.labels_[idx]
with open('word_clust_dict.pkl', 'wb') as of:
    pickle.dump(word_dict, of)

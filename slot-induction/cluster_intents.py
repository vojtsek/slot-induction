from induct import Embedding
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import assert_all_finite

import numpy as np
import json
import spacy
import pickle
import sys
import glob
import argparse

frames_vocab = {}
pos_vocab = {}
word_vocab = {}
stopwords = set(stopwords.words('english'))

class Sentence:

    def __init__(self, sentence, analysis, embeddings, topic, sdir, no, intent, clust_dict):
        self.topic = topic
        self.intent = intent
        self.dir = sdir
        self.no = no
        self.pos = [str(entry[2]) for entry in analysis]
        self.tokens = [str(entry[0]) for entry in analysis if str(entry[0]) not in stopwords]
        self.boc = [0] * 10
        for tk in self.tokens:
            tk = tk.lower()
            if tk in clust_dict:
                self.boc[clust_dict[tk]] += 1
        self.embeddings = embeddings
        self.bow = self.bag_of_words()
        self.bop = self.bag_of_pos()
        self.boe = self.bag_of_embeddings()

    def _to_oh(self, tk, vocab):
        #res = np.zeros((len(self.frames_vocab)))
        res = np.zeros((len(vocab)))
        res[vocab[tk]] = 1
        return res

    def bag_of_embeddings(self):
        return np.mean([self.embeddings[tk] for tk in self.tokens if tk not in stopwords], axis=0)

    def bag_of_words(self):
        return np.sum([self._to_oh(word, word_vocab) for word in self.tokens], axis=0)

    def bag_of_pos(self):
        return np.sum([self._to_oh(pos, pos_vocab) for pos in self.pos], axis=0)

    @property
    def features(self):
        return self.bag_of_frames()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--features_file', required=True)
    args = parser.parse_args()
    file_list = sorted(glob.glob(args.root + '/*'))
    with open('top10kws.pkl', 'rb') as inf:
        word_clust_disct = pickle.load(inf)
    if args.features:
        nlp = spacy.load("en_core_web_sm")
        embeddings = Embedding(args.embeddings)
    def get_analyze_gen(construct=False):
        for f in file_list:
            with open(f + '/raw.txt', 'rt') as inf,\
                 open(f + '/topics.txt') as topic_f,\
                 open(f + '/state.json') as state_f:
                    idx = 0
                    state = {}
                    for line, top in zip(inf, topic_f):
                        topic = [float(t) for t in top.split()]
                        doc = nlp(line)
                        state_line = state_f.readline().strip()
                        if len(state_line) == 0:
                            continue
                        state_gt = json.loads(state_line)
                        intent = None
                        #if len(state_gt['slots']) > 0:
                        #    intent = 'inform'
                        #elif any(state_gt['requested'].values()):
                         #   intent = 'request'
                        for slot, val in state_gt.items():
                            if slot not in state and val != 'not mentioned':
                                state[slot] = val
                                if val in ['yes', 'no']:
                                    intent = 'request'
                                else:
                                    intent = 'inform'
#                                if slot in ['pricerange','food','area']:
#                                    intent = 'inform'
#                                else:
#                                    intent = 'request'
#    for i, doc in enumerate(nlp.pipe(text_gen, n_threads=1, batch_size=1)):
#        for sent in doc.sents:
                        analysis = []
                        for s in doc.sents:
                            for w in s:
                                analysis.append((w, w.lemma_, w.pos_, w.ent_type_))
                                if w.pos_ not in pos_vocab:
                                    pos_vocab[w.pos_] = len(pos_vocab)
                                if w.text not in word_vocab and str(w.text) not in stopwords:
                                    word_vocab[str(w.text)] = len(word_vocab)
                        if construct:
                            idx += 1
                            print(idx)
                            yield Sentence(line, analysis, embeddings, topic, f, idx, intent, word_clust_disct)
                        else:
                            yield False

    if args.features:
        list(get_analyze_gen())
        all_sents = list(get_analyze_gen(True))
        with open(args.features_file, 'wb') as f:
            pickle.dump(all_sents, f)
    else:
        with open(args.features_file, 'rb') as f:
            all_sents = pickle.load(f)
        all_sents_topic = np.array([sent.topic for sent in all_sents])
        all_sents_words = normalize(np.array([sent.bow_enc for sent in all_sents]))
        all_sents_pos = normalize(np.array([sent.bop_enc for sent in all_sents]))
        all_sents_embs = normalize(np.array([sent.boe_enc for sent in all_sents]))
        all_sents_clust = normalize(np.array([sent.boc for sent in all_sents]))
        for n, t in enumerate(all_sents_topic):
            if not np.all(np.isfinite(t)):
                all_sents_topic[n] = np.ones((all_sents_topic.shape[1],)) / all_sents_topic.shape[1]
        all_sents_np = np.concatenate((all_sents_topic, all_sents_words, all_sents_pos), axis=1)
#all_sents = [' '.join(sent.tokens) for sent in all_sents]
#vectorizer = TfidfVectorizer(tokenizer=lambda t: [stemmer.stem(tk) for tk in word_tokenize(t)],
#                             stop_words=stopwords,
#                             lowercase=True)
#tfidf = vectorizer.fit_transform(all_sents)
#    clustering = KMeans(n_clusters=3)
        clustering = SpectralClustering(n_clusters=3, affinity='rbf')
#    clustering = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='cosine')
        clustering = clustering.fit(all_sents_np)
        for current_idx, sent in enumerate(all_sents):
            print(sent.dir, sent.no, 'clust' + str(clustering.labels_[current_idx]))
        #print(sent.intent, 'clust' + str(clustering.labels_[current_idx]))

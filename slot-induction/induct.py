import json
import glob
import sys
import itertools
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import csv
from collections import defaultdict

clf = LocalOutlierFactor(n_neighbors=3, contamination=0.2, metric='cosine', algorithm='brute')

global_sims = defaultdict(lambda: defaultdict(float))
vocab = set()

class Embedding:
    def __init__(self, emb_file):
        self.matrix = pd.read_csv(emb_file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)

    def __getitem__(self, item):
        if item in self.matrix.index:
            return self.matrix.loc[item].values
        return np.zeros(len(self.matrix.columns))
    
    def items(self):
        for item in self.matrix.index:
            yield item, self.matrix.loc[item].values

    def contains(self, item):
        return item in self.matrix.index


class SemanticSlot:
    def __init__(self, name, embeddings, ngh_repr):
        self.name = name
        self.spans = []
        self.distinct_spans = set()
        self.coherence = None
        self.occurences = 0
        self.frequency = 0
        self.embeddings = embeddings
        self.neighbor_based_representation = ngh_repr

    def add_span(self, text):
        self.spans.append(text)
        self.occurences += 1
    
    def compute_frequency(self, total_slots):
        self.frequency = self.occurences / total_slots

    def _embed(self, span, embeddings):
        embedding = np.mean(np.stack([embeddings[tk] for tk in span], axis=0), axis=0)
        return embedding

    def _compute_neighbor_sim(self, span1, span2):
        span1 = self.neighbor_based_representation[span1]
        span2 = self.neighbor_based_representation[span2]
#        print('Span1', span1)
#        print('Span2', span2)
#        print('dot', np.dot(span1, span2))
#        print('norm1', np.linalg.norm(span1))
#        print('norm2', np.linalg.norm(span2))
        #sim = np.dot(span1, span2) / (np.linalg.norm(span1) * np.linalg.norm(span2))
        sim = np.linalg.norm(span1 - span2) / len(span1)
        if np.isnan(sim):
            sim = 0
        return sim

    def _compute_sim(self, span1, span2):
        full1 = span1
        full2 = span2
        span1_tk = word_tokenize(span1)
        span2_tk = word_tokenize(span2)
        sims = []
        for embeddings in self.embeddings:
            if any([not embeddings.contains(tk) for tk in span1_tk + span2_tk]):
                sims.append(-2)
                continue
            span1 = self._embed(span1_tk, embeddings)
            span2 = self._embed(span2_tk, embeddings)
#        print('Span1', span1)
#        print('Span2', span2)
#        print('dot', np.dot(span1, span2))
#        print('norm1', np.linalg.norm(span1))
#        print('norm2', np.linalg.norm(span2))
            sim = np.dot(span1, span2) / (np.linalg.norm(span1) * np.linalg.norm(span2))
            if np.isnan(sim):
                sim = 0
            sims.append(sim)
        vocab.add(full1)
        vocab.add(full2)
        global_sims[full1][full2] = np.mean(sims)
        global_sims[full2][full1] = np.mean(sims)
        return sims

    def filter_outliers(self):
#        X = np.array([emb for _, emb in self.embedded])
#        pred = clf.fit_predict(X)
#        print([(w[0], pr) for w, pr in zip(self.embedded, pred)])

        if len(self.distinct_spans) < 3:
            return
        orig = self.compute_coherence()[0]
        to_omit = set()
        for chosen in self.distinct_spans:
            if len(self.distinct_spans) < 3:
                return
            leave_chosen = [sp for sp in self.distinct_spans if sp != chosen]
            coh_1out = self.compute_coherence(spans=leave_chosen)[0]
            relative_improvement = (coh_1out - orig) * (len(self.distinct_spans) - 1)
            if relative_improvement > 0.5:
                to_omit.add(chosen)
        self.distinct_spans = self.distinct_spans.difference(to_omit)

    def compute_coherence(self, neigh=False, spans=None):
        if spans is None:
            spans = self.distinct_spans
        if len(spans) < 2:
            self.coherences = [0, 0]
            self.neigh_coherence = 0
            return -3
        # for span1, span2 in itertools.combinations(self.distinct_spans, 2):
        #    print(span1, span2)
        #    print(self._compute_sim(span1, span2))
        comb = list(itertools.combinations(spans, 2))
        if neigh:
            self.neigh_coherence = sum([self._compute_neighbor_sim(span1, span2) for span1, span2 in comb]) / len(comb)
        else:
            similarities = np.array([self._compute_sim(span1, span2) for span1, span2 in comb])
            self.coherences = np.sum(similarities, axis=0) / len(comb)
        return self.coherences

    def finalize(self):
        self.distinct_spans = set([self.filter_by_embeddings(sp) for sp in set(self.spans) if len(self.filter_by_embeddings(sp)) > 0])
        # self.embedded = [(sp, self.embeddings[sp]) for sp in self.distinct_spans]
        self.filter_outliers()

    def filter_by_embeddings(self, span):
        return ' '.join([w for w in word_tokenize(span) if self.embeddings[1].contains(w)])


class InducedSlots:
    def __init__(self, embeddings, alpha):
        self.slot_dict = {}
        self.slots_seen = 0
        self.embeddings = embeddings
        self.alpha = alpha
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.neighbor_based_representation = {}

    def process_slot(self, slot_name, slot_span):
        if slot_name not in self.slot_dict:
            self.slot_dict[slot_name] = SemanticSlot(slot_name, self.embeddings, self.neighbor_based_representation)
        tokenized = word_tokenize(slot_span)
        if len(tokenized) < 4 and all([not tk.isdigit() for tk in tokenized]):
            normalized = self.normalize(slot_span)
            if len(normalized) > 0:
                self.slot_dict[slot_name].add_span(normalized)
        self.slots_seen += 1

    def normalize(self, text):
        text = word_tokenize(text)
        text = [self.lemmatizer.lemmatize(w.lower()) for w in text]
        text = [w for w in text if w not in self.stopwords]
        text = ' '.join(text)
        return text

    def _compute_neighbor_representations(self):
        for span in vocab:
            others = global_sims[span].items()
            sorted_others = sorted(others, reverse=True, key=lambda x: x[1])[:10]
            picked = [o[0] for o in sorted_others]
            span_vec = []
            for k, v in others:
                if k in picked:
                    # span_vec.append(v)
                    span_vec.append(1)
                else:
                    span_vec.append(0)
            self.neighbor_based_representation[span] = np.array(span_vec)
 
    def finalize(self):
        for slot in self.slot_dict.values():
            slot.finalize()
            slot.compute_frequency(self.slots_seen)
            slot.compute_coherence()
        comb = list(itertools.combinations(vocab, 2))
        i=0
        for c in comb:
            i += 1
            slot._compute_sim(c[0], c[1])
        self._compute_neighbor_representations()
        for slot in self.slot_dict.values():
            slot.compute_coherence(neigh=True)

    @property
    def slots(self):
        return self.slot_dict.values()

    def _rank_f(self, slot):
        slot = slot[1]
        if slot.coherences is None:
            return 0
        return (1 - alpha) * slot.frequency + alpha * slot.coherences[0]

    @property
    def sorted_slots(self):
        return sorted(self.slot_dict.items(), key=self._rank_f)


if __name__ == '__main__':
    alpha = 0
    embeddings_elmo = Embedding(sys.argv[2])
    embeddings_cn = Embedding('~/hudecek/data/numberbatch/numberbatch-en-17.06.txt')
    induced_slots = InducedSlots([embeddings_elmo, embeddings_cn], alpha)
    of = open(sys.argv[3], 'wt')
    for dial_dir in glob.glob(sys.argv[1] + '/*'):
        state = {}
        with open(dial_dir + '/frames.json', 'rt') as pred_f:
       #     for state_line, frames_line in zip(state_f, pred_f):
            for frames_line in pred_f:
                delta_vector = []
                # state_line = json.loads(state_line)['semi']
                try:
                    #state_line = json.loads(state_line)
                    # state_line = json.loads(state_line)['slots']
                    predicted = json.loads(frames_line)
                    print(predicted)
                    for slot_name, slot_span in [(k, v) for k, v in  predicted.items() if '-score' not in k]:
        #            for slott in predicted:
                     #   for slot_name, slot_span in slott.items():
                        induced_slots.process_slot(slot_name, slot_span)
                except Exception as e:
                    print(e)
#    while True:
#        slot = input('Please input name slot: ')
#        if slot in induced_slots.slot_dict:
#            slot = induced_slots.slot_dict[slot]
#            slot.finalize()
#            slot.compute_frequency(induced_slots.slots_seen)
#            slot.compute_coherence()
#        else:
#            print('Unknown slot')
    induced_slots.finalize()
    for slot_name, slot in induced_slots.sorted_slots:
        print('{} {} {} {} {}'.format(slot_name, slot.frequency, ' '.join([str(c) for c in slot.coherences]), slot.neigh_coherence, json.dumps(list(slot.distinct_spans))), file=of)
    of.close()
     

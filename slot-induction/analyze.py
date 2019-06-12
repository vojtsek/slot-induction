import sys
import glob
import math
import json
import os
import numpy as np
from itertools import combinations
from collections import defaultdict
from induct import Embedding
import argparse
import pickle
import re
from nltk import word_tokenize, pos_tag

class InductedSlot:
    def __init__(self, name, coherence_table):
        self.name = name
        self.ctx_size = 2
        self.frame_names = []
        self.total = 0
        self.threshold = 0.05
        self.templates = []
        self.coherence_table = coherence_table
        self.all_templates_counts = defaultdict(int)

    def add_frame(self, name):
        self.frame_names.append(name)

    def matches(self, parse, sent):
        val = self.matches_parse(parse)
        if val is not None:
            return ' '.join(val)
        for tmpl in self.templates:
            mtch = tmpl.match(sent)
            if mtch is not None:
                return mtch.group('slotval')
        return None

    
    def matches_parse(self, parse):
        for key in self.frame_names:
            if key in parse:
                val = parse[key]
                val_pos = pos_tag(word_tokenize(val))
                val = [w[0] for w in val_pos if w[1] not in ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO'] and 'VB' not in w[1]]
                val = [w.lower() for w in val if w.lower() not in ['where', 'thanks', 'thank', 'yes', 'no', 'there', '?', '!', '.', 'of', 'a', ',', '"', '\'', 'the']]
                if len(val) > 0 and len(val) < 4:
                    return val
        return None

    def read_sentence(self, sent, parse):
        val = self.matches_parse(parse)
        sent = word_tokenize(sent.lower())
        if val is not None and val[0] in sent:
            idx = sent.index(val[0])
            left_idx = max(0, idx - self.ctx_size)
            right_idx = min(len(sent), idx + self.ctx_size)
            left_ctx = sent[left_idx:idx] if idx > 0 else []
            right_ctx = sent[(idx+1):right_idx] if idx < len(sent) - 1 else []
            tmpl = '.*' + ' '.join(left_ctx + ['(?P<slotval>\w+)'] + right_ctx) + '.*'
            self.total += 1
            self.all_templates_counts[tmpl] += 1

    def calculate(self):
        self.templates = [re.compile(k) for k, v in\
                          sorted(self.all_templates_counts.items(), key=lambda x: x[1] / self.total, reverse=True) if v / self.total > self.threshold]

        # print(sorted(self.all_templates_counts.items(), key=lambda x: x[1] / self.total,reverse=True))
        self.frame_names = sorted(self.frame_names, key=lambda x: self.coherence_table[x], reverse=True)
        print(self.templates)

class Slot:
    def __init__(self, slot_name, frq, coher, neigh_coherence, fillers, topic, embeddings):
        self.slot_name = slot_name
        self.topic = topic
        self.frequency = frq
        self.coherence = coher
        self.neigh_coherence = neigh_coherence
        self.embeddings = embeddings
        self.fillers = json.loads(' '.join(fillers))
        self.embedded = None
        self.rankings = []

    def embed(self):
        fillers = [tk for tkns in self.fillers for tk in tkns.split()if self.embeddings.contains(tk)]
        if len(fillers) > 0:
            self.embedded = np.mean(np.stack([self.embeddings[tk] for tk in fillers]), axis=0)
        else:
            self.embedded = self.embeddings['<UNK>']

    def similarity(self, other):
        set1 = set([tk for tkns in self.fillers for tk in tkns.split()])
        set2 = set([tk for tkns in other.fillers for tk in tkns.split()])
        if len(set1) == 0 or len(set2) == 0:
            return 0
        return len(set1.intersection(set2)) / min(len(set1), len(set2))
#        if self.embedded is None:
#            self.embed()
#        if other.embedded is None:
#            other.embed()
#        return np.dot(self.embedded, other.embedded) /\
#               (np.linalg.norm(self.embedded) * np.linalg.norm(self.embedded))
#
slot_list = []

def get_rank_f(alpha):
    def f(slot):
        if slot.coherence is None:
            return 0
        return alpha * slot.frequency + (1 - alpha) * slot.coherence
    return f


class UnsupervisedRanker:
# http://kevinsmall.org/pdf/KlementievRoSm07.pdf
    
    def __init__(self, f_rankings, threshold, lmb=.1):
        self.f_rankings = f_rankings
        self.weights = [0 for _ in f_rankings]
        self.threshold = threshold
        self.lmb = lmb

    def _nx(self, x):
        nx = [f for f in self.f_rankings if f[x] <= self.threshold]
        return nx

    def _mean(self, x, nx):
        if len(nx) > 0:
            mean = sum([f[x] for f in nx]) / len(nx)
        else:
            mean = self.threshold
        return mean

    def estimate_weights(self):
        for x_index in range(len(self.f_rankings[0])):
            n_x = self._nx(x_index)
            mean_x = self._mean(x_index, n_x)
            for n, f in enumerate(self.f_rankings):
                if f[x_index] <= self.threshold:
                    delta = np.power(f[x_index] - mean_x, 2)
                else:
                    delta = np.power(self.threshold + 1 - mean_x, 2)
                self.weights[n] += self.lmb * delta
        print(np.array(self.weights) / np.sum(self.weights))

    def rank(self):
        ranks = []
        for x_index in range(len(self.f_rankings[0])):
            ranks.append(0)
            for n, f in enumerate(self.f_rankings):
                if f[x_index] <= self.threshold:
                    ranks[-1] += self.weights[n] * f[x_index]
                else:
                    ranks[-1] += self.weights[n] * self.threshold
        return ranks


if __name__ == '__main__':
    embeddings = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_file', required=True)
    parser.add_argument('--similarities', required=True)
    parser.add_argument('--mode', default='analyze')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--topics', type=int, default=10)
    parser.add_argument('--output_mapping', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    with open(args.frame_file, 'rt') as inf:
        for line in inf:
            line = line.split()
            if 'nan' in line[0]:
                topic=-1
            else:
                topics = [float(t) for t in line[:args.topics]]
                topic = np.argmax(topics)
            line = line[args.topics:]
            frq = float(line[1].strip(',')) if line[1] != 'None' else None
            coh = float(line[2].strip(',')) if line[2] != 'None' else None
            ngh_coh = float(line[3].strip(',')) if line[3] != 'None' else None
            if len(line[4:]) < 3:
                continue
            slot = Slot(line[0].strip(','), frq, coh, ngh_coh, line[4:], topic, embeddings)
            slot_list.append(slot)

    if args.mode == 'sim':
        similarities = defaultdict(dict)
        for sl1, sl2 in combinations(slot_list, 2):
            threshold = 0.005
            if sl1.frequency < threshold or sl2.frequency < threshold:
                continue
            sim = sl1.similarity(sl2)
            similarities[sl1.slot_name][sl2.slot_name] = sim
            similarities[sl2.slot_name][sl1.slot_name] = sim

        with open(args.similarities, 'wt') as of:
            json.dump(similarities, of)
    else:
        with open(args.similarities, 'rt') as f:
            similarities = json.load(f)

        # sorted_list = sorted(slot_list, key=get_rank_f(args.alpha))
        for n, sl in  enumerate(sorted(slot_list, key=lambda sl: sl.frequency, reverse=True)):
            sl.rankings.append(n)
        for n, sl in  enumerate(sorted(slot_list, key=lambda sl: sl.coherence, reverse=True)):
            sl.rankings.append(n)
        for n, sl in  enumerate(sorted(slot_list, key=lambda sl: sl.neigh_coherence, reverse=True)):
            sl.rankings.append(n)

        rankings = [[], [], []]
        for sl in slot_list:
            rankings[0].append(sl.rankings[0])
            rankings[1].append(sl.rankings[1])
            rankings[2].append(sl.rankings[2])
        ranker = UnsupervisedRanker(rankings, threshold=25, lmb=0.9)
        ranker.estimate_weights()
        ranks = ranker.rank()
        sorted_list = [sl for _, sl in sorted(zip(ranks, slot_list), key=lambda x: x[0], reverse=True)]
        clusters = {}
        current_cluster = 0
        coherence_table = {}
        for sl in sorted_list:
            print(sl.slot_name, sl.topic, sl.frequency, sl.coherence, sl.neigh_coherence, sl.rankings, sl.fillers)
            coherence_table[sl.slot_name] = sl.coherence
        chosen = sorted_list[-15:]
        for sl in chosen:
            slot = sl.slot_name
            chosen_names = [sl.slot_name for sl in chosen]
            same_topic_slots = [similar.slot_name for similar in chosen if similar.topic == sl.topic]
            if slot not in clusters:
                clusters[slot] = current_cluster
                for same_topic_slot in same_topic_slots:
                    clusters[same_topic_slot] = current_cluster
            current_cluster += 1
        inducted = {}
        for val in set(clusters.values()):
            inducted[val] = InductedSlot(str(val), coherence_table)
        for frame_name, cl in clusters.items():
            inducted[cl].add_frame(frame_name)
        for sl in inducted.values():
            for d in glob.glob(args.root + '/*'):
                if not os.path.exists(d + '/semafor-frames.json'):
                    continue
                with open(d + '/raw.txt', 'rt') as raw_f,\
                     open(d + '/semafor-frames.json', 'rt') as frame_f:
                        for raw, frames in zip(raw_f, frame_f):
                            frames = json.loads(frames)
                            sl.read_sentence(raw, frames)
            sl.calculate()
            print(sl.frame_names)
    with open(args.output_mapping, 'wb') as of:
        pickle.dump(inducted, of)

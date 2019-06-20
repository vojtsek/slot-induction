import sys
import os
import glob
import argparse
import pickle
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import json
import itertools
from copy import deepcopy

from cluster_intents import Sentence
from induct import Embedding
from analyze import Slot, InductedSlot
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


def cos_loss(x, y):
    return keras.backend.abs(keras.losses.cosine_proximity(x, y))

class ClusterNode:
    def __init__(self, value, order):
        self.value = value
        self.order = order
        self.children = []
        self.parent = None

    def get_values(self):
        if len(self.children) == 0:
            return [self.value]
        return self.children[0].get_values() + self.children[1].get_values()

    def get_orders(self):
        if len(self.children) == 0:
            return [self.order]
        return self.children[0].get_orders() + self.children[1].get_orders()

class Clustering:
    def __init__(self, data, distance):
        self.distance = distance
        self.labels = []
        self.current_clusters = [ClusterNode(d, n) for n, d in enumerate(data)]

    def fit(self):
        n_clusters = len(self.current_clusters)
        while n_clusters > 1:
            n_clusters -= 1
            merge_candidates = self._get_merge_candidates()
            self.current_clusters = [c for c in self.current_clusters if c not in merge_candidates]
            merged = self._merge(merge_candidates)
            self.current_clusters.append(merged)

    def _get_merge_candidates(self):
        lowest_distance = 1 / 0.00001
        candidates = None
        for comb in itertools.combinations(self.current_clusters, 2):
            dist = self.distance(*comb)
            if dist < lowest_distance:
                lowest_distance = dist
                candidates = comb
        return candidates

    def _merge(self, nodes):
        merged = ClusterNode(None, None)
        merged.children = nodes
        for node in nodes:
            node.parent = merged
        return merged

def get_distance(d, linkage='avg'):
    def f(cluster1, cluster2):
        vals1 = cluster1.get_values()
        vals2 = cluster2.get_values()
        distances = []
        for v1 in vals1:
            for v2 in vals2:
                distances.append(d(v1, v2))
        if linkage == 'avg':
            return np.mean(distances)
    return f


def avg_dissim_within_group_element(ele, element_list):
    max_diameter = -np.inf
    sum_dissm = 0
    for i in element_list:
        sum_dissm += dissimilarity_matrix[ele][i]   
        if( dissimilarity_matrix[ele][i]  > max_diameter):
            max_diameter = dissimilarity_matrix[ele][i]
    if(len(element_list)>1):
        avg = sum_dissm/(len(element_list)-1)
    else: 
        avg = 0
    return avg

def avg_dissim_across_group_element(ele, main_list, splinter_list):
    if len(splinter_list) == 0:
        return 0
    sum_dissm = 0
    for j in splinter_list:
        sum_dissm = sum_dissm + dissimilarity_matrix[ele][j]
    avg = sum_dissm/(len(splinter_list))
    return avg
    
    
def splinter(main_list, splinter_group):
    most_dissm_object_value = -np.inf
    most_dissm_object_index = None
    for ele in main_list:
        x = avg_dissim_within_group_element(ele, main_list)
        y = avg_dissim_across_group_element(ele, main_list, splinter_group)
        diff= x -y
        if diff > most_dissm_object_value:
            most_dissm_object_value = diff
            most_dissm_object_index = ele
    if(most_dissm_object_value>0):
        return  (most_dissm_object_index, 1)
    else:
        return (-1, -1)
    
def split(element_list):
    main_list = element_list
    splinter_group = []    
    (most_dissm_object_index,flag) = splinter(main_list, splinter_group)
    while(flag > 0):
        main_list.remove(most_dissm_object_index)
        splinter_group.append(most_dissm_object_index)
        (most_dissm_object_index,flag) = splinter(element_list, splinter_group)
    
    return (main_list, splinter_group)

def max_diameter(cluster_list):
    max_diameter_cluster_index = None
    max_diameter_cluster_value = -np.inf
    index = 0
    for element_list in cluster_list:
        for i in element_list:
            for j in element_list:
                if dissimilarity_matrix[i][j]  > max_diameter_cluster_value:
                    max_diameter_cluster_value = dissimilarity_matrix[i][j]
                    max_diameter_cluster_index = index
        
        index +=1
    
    if(max_diameter_cluster_value <= 0):
        return -1
    
    return max_diameter_cluster_index
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_file', required=True)
    parser.add_argument('--feature', type=str)
    parser.add_argument('--output')
    parser.add_argument('--input')
    parser.add_argument('--output_mapping')
    parser.add_argument('--root')
    parser.add_argument('--no_clusters', type=int, default=6)
    parser.add_argument('--encoded_dim', type=int)
    # parser.add_argument('--sentece_pkl', required=True)
    args = parser.parse_args()

    # embeddings = Embedding('~/hudecek/data/numberbatch/numberbatch-en-17.06.txt')
    with open(args.frame_file, 'rb') as inf:
        frames = pickle.load(inf)
    coherence_table = {}
    for frame in frames:
        coherence_table[frame.slot_name] = frame.coherence1

    for feat in ['embedded', 'pos_feature']:
        x_data = np.array([getattr(fr, feat) for fr in frames], dtype='float64')
        x_data /= np.max(x_data)
        in_dim = x_data.shape[1]
        input_l = keras.layers.Input(shape=(in_dim,))
        encoded = keras.layers.Dense(args.encoded_dim,)(input_l)
        #encoded = keras.layers.Dense(args.encoded_dim, activation='relu', activity_regularizer=keras.regularizers.l1(10e-5))(input_l)
        reconstructed = keras.layers.Dense(in_dim, activation='sigmoid')(encoded)
        ae_model = keras.models.Model(input_l, reconstructed)
        encoder = keras.models.Model(input_l, encoded)
        ae_model.compile(optimizer='adadelta', loss='binary_crossentropy')
        ae_model.fit(x_data, x_data, epochs=200, shuffle=True)
        encoded = encoder.predict(x_data)
        for i, fr in enumerate(frames):
            setattr(fr, feat + '_enc', encoded[i])

    frames_topic = np.array([frame.topics for frame in frames])
    frames_words = normalize(np.array([frame.embedded_enc for frame in frames]))
    frames_pos = normalize(np.array([frame.pos_feature_enc for frame in frames]))
    for n, t in enumerate(frames_topic):
        if not np.all(np.isfinite(t)):
            frames_topic[n] = np.ones((frames_topic.shape[1],)) / frames_topic.shape[1]
    frames_np = np.concatenate((frames_topic, frames_words, frames_pos), axis=1)
#

    clustering = Clustering(frames_np, get_distance(lambda a, b: np.linalg.norm(a-b)))
    clustering.fit()
    root = clustering.current_clusters[0]
    print([[frames[o].slot_name for o in c.get_orders()] for c in root.children])

    all_elements_orig = [f.slot_name for f in frames]
    all_elements = [f.slot_name for f in frames]
    current_clusters = ([all_elements])
#    frames_np = frames_topic
    level = 1
    index = 0
    while(index!=-1 and level != args.no_clusters):
        diss = []
        distance = get_distance(lambda a, b: np.linalg.norm(a-b))
        for f1 in frames_np:
            diss.append([np.linalg.norm(f1 - f2) for f2 in frames_np])
        mat = np.array(diss)
        dissimilarity_matrix = pd.DataFrame(mat,index=all_elements_orig, columns=all_elements_orig)
        for c in current_clusters:
            print(level, c)
        (a_clstr, b_clstr) = split(current_clusters[index])
        del current_clusters[index]
        current_clusters.append(a_clstr)
        current_clusters.append(b_clstr)
        index = max_diameter(current_clusters)
        level +=1
        print('-' * 80)

    inducted = {}
    for n, c in enumerate(current_clusters):
        print(level, c)
        inducted[n] = InductedSlot(str(n), coherence_table)
        for fr in c:
            inducted[n].add_frame(fr)
        for sl in inducted.values():
            for d in glob.glob(args.root + '/*'):
                if not os.path.exists(d + '/frames.json'):
                    continue
                with open(d + '/raw.txt', 'rt') as raw_f,\
                     open(d + '/frames.json', 'rt') as frame_f:
                        for raw, frames in zip(raw_f, frame_f):
                            frames = json.loads(frames)
                            sl.read_sentence(raw, frames)
            sl.calculate()
            print(sl.frame_names)
    with open(args.output_mapping, 'wb') as of:
        pickle.dump(inducted, of)

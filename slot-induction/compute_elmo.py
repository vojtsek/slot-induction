import sys
import glob
from elmoformanylangs import Embedder
from nltk import word_tokenize
import numpy as np
from collections import defaultdict

embedder = Embedder(sys.argv[2])
all_words_dict = defaultdict(list)
sents = []
for f in glob.glob(sys.argv[1] + '/*'):
    with open(f + '/raw.txt', 'rt') as inf:
        for line in inf:
            sents.append(word_tokenize(line))

embedded = embedder.sents2elmo(sents)
for tokenized, embedded_s in zip(sents, embedded):
    for word, emb in zip(tokenized, embedded_s):
        all_words_dict[word].append(emb)

with open(sys.argv[3], 'wt') as of:
    for tk, embs in all_words_dict.items():
        avg_emb = np.mean(np.stack(embs), axis=0)
        print(tk, ' '.join([str(e) for e in avg_emb]), file=of)


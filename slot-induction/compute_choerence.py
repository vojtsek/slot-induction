from induct import Embedding, SemanticSlot


#embeddings = Embedding('~/hudecek/data/word2vec/wiki-news-300d-1M.vec')
embeddings = Embedding('~/hudecek/data/elmo/elmo_camrest_embs.txt')
with open('sample-slot.txt', 'rt') as f: 
    for line in f:
        s = SemanticSlot('x', embeddings)
        line = line.split()
        s.spans = line
        s.finalize()
        s.compute_coherence()
        print(s.coherence)

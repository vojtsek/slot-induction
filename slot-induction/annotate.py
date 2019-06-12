import argparse
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from functools import reduce
import pickle
import glob
import json

from analyze import InductedSlot

lemmatizer = WordNetLemmatizer()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot_file', required=True)
    parser.add_argument('--keywords', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()

    with open(args.slot_file, 'rb') as inf:
        slots = pickle.load(inf)
    with open(args.keywords, 'rb') as inf:
        keywords = pickle.load(inf)
 #   keywords['moderate'] = 10
#    keywords['moderately'] = 11
    for d in glob.glob(args.root + '/*'):
        with open(d + '/semafor-frames.json', 'rt') as frame_f,\
             open(d + '/raw.txt', 'rt') as raw_f,\
             open(d + '/state.json', 'rt') as gt_f,\
             open(d + '/predicted-state.json', 'wt') as state_f:
            state = {}
            state_keys = set()
            for turn, text in zip(frame_f, raw_f):
                state['req'] = ','.join([w for w in word_tokenize(text) if w.lower() in keywords])
                detected = json.loads(turn)
                gtline = gt_f.readline()
                if len(gtline) == 0:
                    gt = {}
                else:
                    gt = json.loads(gtline)
                #for fr, val in detected.items():
                #    if fr in slots:
                #        val_pos = pos_tag(word_tokenize(val))
                #        val = [w[0] for w in val_pos if w[1] not in ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO'] and 'VB' not in w[1]]
                #        val = [w.lower() for w in val if w.lower() not in ['where', 'thanks', 'thank', 'yes', 'no', 'there', '?', '!', '.', 'of', 'a', ',', '"', '\'', 'the']]
                #        if len(val) > 0 and len(val) < 4:
                #            state[slots[fr]] = ' '.join(val)
                #            state[fr + '-score'] = float(detected[fr + '-score']) / 100
                #            state_keys.add(fr + '-score')
                for sl in slots.values():
                    val = sl.matches(detected, text)
                    if val is not None:
                        state[sl.name] = lemmatizer.lemmatize(val)
                state['score'] = reduce(lambda acc, x: x * acc, [float(state[idx]) for idx in state_keys], 1)
                json.dump(state, state_f)
                print(file=state_f)

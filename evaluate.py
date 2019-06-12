import glob
import sys
import json
from collections import namedtuple

class EvalEntry:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)


class StateItem:
    def __init__(self, slotname, frames):
        self.recognized = 0
        self.gts = 0
        self.slotname = slotname
        self.frames = frames
    
    def test_match(self, predicted_frames, val):
        self.gts += 1
        for pred_f, pred_val in predicted_frames.items():
            if val == 'centre':
                val = 'center'
            if pred_f in self.frames and val.lower() in pred_val.lower():
                self.recognized += 1
                return True
        return False

    def eval(self):
        return '{0:.2f}'.format(self.recognized / self.gts)


all_frames = ['Expensiveness', 'Path', 'Type', 'Locative_relation', 'Direction', 'Origin', 'Goal', 'Locale', 'Whole', 'Desirability', 'Judgment_direct_address']

oovs = {'Whole': 0, 'Desirability': 0, 'Judgment_direct_address': 0, 'Locale': 0}
eval_dir = {'pricerange': StateItem('price', ['Expensiveness']),
            #'area': StateItem('area', ['Part_whole', 'Part', 'Part_inner_outer', 'Locative_relation', 'Direction', 'Part_orientational', 'Locale', 'Locale_by_use']),
            'area': StateItem('area', ['Path','Locative_relation', 'Direction']),
            'food': StateItem('food', ['Type', 'Origin', 'Goal'])}

measure_dir = {'pricerange': EvalEntry(),
               'area': EvalEntry(),
               'food': EvalEntry()}

utts_total = 0
for dial_dir in glob.glob(sys.argv[1] + '/*'):
    state = {}
    print('-'*80)
    with open(dial_dir + '/state.json', 'rt') as state_f, open(dial_dir + '/semafor-frames.json', 'rt') as pred_f, open(dial_dir + '/raw.txt', 'rt') as utt_f:
        for state_line, frames_line,  utt_line in zip(state_f, pred_f, utt_f):
            utts_total += 1
            delta_vector = []
            # state_line = json.loads(state_line)['semi']
            state_line = json.loads(state_line)
            # state_line = json.loads(state_line)['slots']
            predicted = json.loads(frames_line)
            print(utt_line.strip())
            print({k:v for k, v in predicted.items() if k in all_frames})
            for slot, val in state_line.items():
                if slot not in state or val != state[slot]:
                    if slot in eval_dir:
                         if eval_dir[slot].test_match(predicted, val):
                             measure_dir[slot].tp += 1
                         else:
                             measure_dir[slot].fn += 1
                    delta_vector.append(slot)
                state[slot] = val
            for pred, val in predicted.items():
                for slotname, slval in state_line.items():
                    if slotname not in eval_dir:
                        continue
                    si = eval_dir[slotname]
                    if pred in si.frames:
                        if slotname not in state_line.keys() or state_line[slotname] not in val:
                            measure_dir[slotname].fp += 1
                        else:
                            measure_dir[slotname].tn += 1
                if not any([pred in si.frames for si in eval_dir.values()]) and pred in all_frames:
                    oovs[pred] += 1

for slot, ev in eval_dir.items():
    print(slot, ev.eval())

for slot, entry in measure_dir.items():
    print('{0}: {1:.2f} {2:.2f} {3:.2f}'.format(slot, entry.precision, entry.recall, entry.f1))

print({k: v/utts_total for k, v in oovs.items()})

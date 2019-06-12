import json
import sys
import glob

class EvalEntry:
    def __init__(self):
        self.tp = 1e-8
        self.fp = 1e-8
        self.tn = 0
        self.fn = 1e-8

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

eval_entries = {'pricerange': EvalEntry(),
                'area': EvalEntry(),
                'food': EvalEntry(),
                'req-address': EvalEntry(),
                'req-phone': EvalEntry(),
                'req-postcode': EvalEntry()}
slot_mapping = {'area': '10', 'pricerange': '8', 'food': '0'}
slot_mapping = {'area': '11', 'pricerange': '10', 'food': '0'}
for d in glob.glob(sys.argv[1] + '/*'):
    with open(d + '/state.json', 'rt') as state_gt_f,\
    open(d + '/predicted-state.json', 'rt') as state_pr_f:
        for line_gt, line_pred in zip(state_gt_f, state_pr_f):
            gt = json.loads(line_gt)
            pred = json.loads(line_pred)
            for item, entry in eval_entries.items():
                if 'req' in item:
                    sl = item[4:]
                    if sl in gt:
                        if sl in pred['req']:
                            entry.tp += 1
                        else:
                            entry.fn += 1
                    elif sl in pred['req']:
                        entry.fp += 1
                else:
                    if item in gt:
                        val = gt[item]
                        pred_val = pred[slot_mapping[item]] if slot_mapping[item] in pred else ''
                        if val in pred_val:
                            entry.tp += 1
                        else:
                            entry.fn += 1
                    elif slot_mapping[item] in pred:
                        entry.fp += 1

for slot, entry in eval_entries.items():
    print('{}: precision {}, recall {}, F-1 {}'.format(slot, entry.precision, entry.recall, entry.f1))


import glob
import sys
import json
import re

from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from itertools import combinations


lemmatizer = WordNetLemmatizer()

def prune_utterances(utterances):
    commons = defaultdict(list)
    for utt1, utt2 in combinations(utterances, r=2):
        if len(utt1) < 3:
            continue
        utt1 = set(utt1.split())
        utt2 = set(utt2.split())
        diff = utt1.difference(utt2)
        if len(diff) == 1 and len(utt1.intersection(utt2)) > 1:
            commons['-'.join(sorted(list(utt1.intersection(utt2))))].append([d for d in diff][0])
    for common, vals in commons.items():
        if len(set(vals)) > 5: # i.e. word in commons occur together with more than 5 different additional words
            for val in set(vals):
                utterances = [utt.replace(' ' + val + ' ', '\w+') for utt in utterances]
    return set(utterances)


class StateItem:
    def __init__(self, slotname):
        self.recognized = 0
        self.gts = 0
        self.slotname = slotname
        self.frames = defaultdict(lambda: [0, []])
        self.matching_predictions = defaultdict(lambda: defaultdict(set))
        self.all_regexes = []

    def eval(self):
        return '{0:.2f}'.format(self.recognized / self.gts)

    def calc_regex(self):
        all_templates = set()
        for matching, vals in self.matching_predictions.items():
            for value, all_utts in vals.items():
                for string in all_utts:
                    if len(string.split()) == 1:
                        continue
                    string = ' '.join([lemmatizer.lemmatize(word) for word in string.split()])
                    #word_regex = ' '.join(['\w+' for _ in value.split()])
                    word_regex = '\w+'
                    string ='(.* )?' +  string.replace(value, '(?P<slotval>{})'.format(word_regex)) + '( .*)?'
                    all_templates.add(string)
        all_templates = prune_utterances(all_templates)
        self.regexes = [re.compile(template) for template in all_templates]

    def test_match(self, predicted_frames, val):
        self.gts += 1
        for pred_f, pred_val in predicted_frames.items():
            if len(pred_val.split()) > 4:
                continue
            if val.lower() in pred_val.lower():
                self.recognized += 1
                self.frames[pred_f][0] += 1
                pred_val = ' '.join([lemmatizer.lemmatize(word.lower()) for word in pred_val.split()])
                regex_string ='(.* )?' +  pred_val.replace(val.lower(), '(?P<slotval>{})'.format('\w+')) + '( .*)?'
                if regex_string not in self.frames[pred_f][1]:
                    self.frames[pred_f][1].append(regex_string)
                self.matching_predictions[pred_f][val].add(pred_val)
                break

    def matches(self, utt):
        for regex in self.all_regexes:
            for frame_predicted, span in utt.items():
                if regex[1] == frame_predicted:
                    match = regex[0].match(span)
                    if match is not None:
                        return match.group('slotval')
#        for frame_predicted, span in utt.items():
#            if not frame_predicted in self.frames:
#                continue
#            counts = self.frames[frame_predicted]
#            for regex in counts[1]:
#                print(self.slotname, regex, span)
#                match = regex.match(span)
#                if match is not None:
#                    return match.group('slotval')
#        return None

    def compile(self):
        for name, frame in self.frames.items():
            if frame[0] > 3:
                for reg in frame[1]:
                    self.all_regexes.append((reg, name))
        self.all_regexes = [(re.compile(reg[0]), reg[1]) for reg in sorted(self.all_regexes, key=lambda x: len(x[0].split()), reverse=True)]


eval_dir = {'pricerange': StateItem('pricerange'),
            'area': StateItem('area'),
            'phone': StateItem('phone'),
            'food': StateItem('food')}

for dial_dir in glob.glob(sys.argv[1] + '/*'):
    state = {}
    with open(dial_dir + '/state.json', 'rt') as state_f, open(dial_dir + '/semafor-frames.json', 'rt') as pred_f:
        for state_line, frames_line in zip(state_f, pred_f):
            delta_vector = []
            # state_line = json.loads(state_line)['semi']
            state_line = json.loads(state_line)
            # state_line = json.loads(state_line)['slots']
            predicted = json.loads(frames_line)
            for slot, val in state_line.items():
                if slot not in state or val != state[slot]:
                    if slot in eval_dir:
                        eval_dir[slot].test_match(predicted, val)
                    delta_vector.append(slot)
                state[slot] = val

for slot, ev in eval_dir.items():
    ev.compile()

for dial_dir in glob.glob(sys.argv[2] + '/*'):
    with open(dial_dir + '/semafor-frames.json', 'rt') as utt_f:
        for line in utt_f:
            predicted = json.loads(line)
            for slot, ev in eval_dir.items():
                slotval = ev.matches(predicted)
                if slotval is not None:
                    print(slot, slotval)

import sys
from collections import defaultdict

class IntentEntry:
    def __init__(self, name):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.name = name
        self.picked = None
        self.predictions = defaultdict(int)

    def pick_class(self):
        max_count = 0
        for predicted, count in self.predictions.items():
            if count > max_count:
                max_count = count
                self.picked = predicted

    def process_prediction(self, name, pred):
        if name == self.name:
            if pred == self.picked:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if pred == self.picked:
                self.fp += 1
            else:
                self.tn += 1

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)


intents = [IntentEntry('inform'), IntentEntry('request'), IntentEntry('None')]
with open(sys.argv[1], 'rt') as f:
    for line in f:
        line = line.split()
        intent = line[0]
        predicted = line[1]
        for intent_gt in intents:
            if intent_gt.name == intent:
                intent_gt.predictions[predicted] += 1
    for intent_gt in intents:
        intent_gt.pick_class() 
    f.seek(0)
    for line in f:
        line = line.split()
        intent = line[0]
        predicted = line[1]
        for intent_gt in intents:
            intent_gt.process_prediction(intent, predicted)
    for intent_gt in intents:
        print(intent_gt.name, intent_gt.precision, intent_gt.recall, intent_gt.f1)


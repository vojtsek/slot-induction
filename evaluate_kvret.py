import glob
import sys
import json

class StateItem:
    def __init__(self, slotname, frames):
        self.recognized = 0
        self.gts = 0
        self.slotname = slotname
        self.frames = frames

    def eval(self):
        return '{0:.2f}'.format(self.recognized / self.gts)

    def test_match(self, predicted_frames, val):
        self.gts += 1
        for pred_f in predicted_frames:
            if any([f in pred_f.keys() for f in self.frames]) and any([val in predval for predval in pred_f.values()]):
                self.recognized += 1
                break


eval_dir = {'address': StateItem('address', []),
            'poi_type': StateItem('poi_type', ['Food', 'Buildings']),
            'distance': StateItem('distance', ['Roadways', 'Arriving']),
            'traffic_info': StateItem('traffic_info', ['Quantified_mass', 'Roadway', 'Avoiding']),
            'event': StateItem('event', ['Intentionally_act', 'Relative_time', 'Discussion']),
            'time': StateItem('time', ['Performers_and_roles']),
            'date': StateItem('date', ['Process_end', 'Calendric_unit', 'Ranked_expectation', 'Relative_time']),
            'room': StateItem('room', ['Discussion', 'Part_inner_outer', 'Buildings', 'Locative_relation']),
            'agenda': StateItem('agenda', ['Manipulation', 'Being_in_control', 'Discussion', 'Social_event', 'Sign_agreement']),
            'party': StateItem('party', ['Kinship']),
            'location': StateItem('location', ['Political_locales', 'Experiencer_obj', 'Locale_by_use']),
            'poi': StateItem('poi', [])}

for dial_dir in glob.glob(sys.argv[1] + '/*'):
    state = {}
    try:
        with open(dial_dir + '/state.json', 'rt') as state_f, open(dial_dir + '/predicted-frames.json', 'rt') as pred_f:
            for state_line, frames_line in zip(state_f, pred_f):
                delta_vector = []
                # state_line = json.loads(state_line)['semi']
                # state_line = json.loads(state_line)
                state_line = json.loads(state_line)['slots']
                predicted = json.loads(frames_line)
                for slot, val in state_line.items():
                    if slot in eval_dir:
                        eval_dir[slot].test_match(predicted, val)
                    state[slot] = val
    except:
        print(dial_dir)
for slot, ev in eval_dir.items():
    print(slot, ev.eval())


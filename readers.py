import json
from collections import namedtuple

dact = namedtuple('dact', 'type slot value')

class Reader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = []
        self._load()

    def _load(self):
        with open(self.data_path, 'rt') as in_fd:
            self.raw_data = json.load(in_fd)

    def call_for_every_turn(self, func, acc):
        for item in self.turns_iter():
            acc = func(acc, item)
        return acc


class CamRestReader(Reader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def dial_iter(self):
        for dial in self.raw_data:
            yield {'id': dial['dialogue_id'],
                   'goal': {'constraints': self._process_slots(dial['goal']['constraints']),
                            'request-slots': dial['goal']['request-slots']},
                   'turns_iterator': self._make_turn_iter(dial['dial'])}

    def turns_iter(self):
        for dial in self.dial_iter():
            for turn in dial['turns_iterator']:
                yield turn

    def _make_turn_iter(self, turns):
        for turn in turns:
            yield {'number': turn['turn'],
                   'usr_trn': turn['usr']['transcript'],
                   'sys_trn': turn['sys']['sent'],
                   'usr_slu': self._process_das(turn['usr']['slu'])}

    def _process_slots(self, slots):
        return {slot_entry[0]: slot_entry[1] for slot_entry in slots}

    def _process_das(self, das):
        return [dact(da['act'], da['slots'][0][0], da['slots'][0][1]) for da in das]


class MultiWOZReader(Reader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def dial_iter(self):
        for did, dial in self.raw_data.items():
            yield {'id': did,
                   'goal': dial['goal'],
                   'turns_iterator': self._make_turn_iter(dial['log'])}

    def turns_iter(self):
        for dial in self.dial_iter():
            for turn in dial['turns_iterator']:
                yield turn

    def _make_turn_iter(self, turns):
        for n, turn in enumerate(turns):
            yield {'number': n,
                   'usr_trn': turn['user'],
                   'sys_trn': turn['system'],
                   'usr_slu': self._process_das(turn['metadata'])}

    def _process_slots(self, slots):
        return {slot_entry[0]: slot_entry[1] for slot_entry in slots}

    def _process_das(self, das):
        return [dact(da['act'], da['slots'][0][0], da['slots'][0][1]) for da in self._convert_log(das)]
    
    def _convert_log(self, das):
        for domain, d_info in das.items():
            for slot_name, slot_value in d_info['semi'].items():
                if len(slot_value) > 0:
                    yield {'act': 'inform',
                           'slots': (('{}-{}'.format(domain, slot_name), slot_value),)}


class MultiWOZRestaurantReader(MultiWOZReader):
    def __init__(self, data_path):
        super().__init__(data_path)
    
    def dial_iter(self):
        for did, dial in self.raw_data.items():
            if len(dial['goal']['restaurant']) > 0 and \
                sum([len(dial['goal'][domain]) for domain in dial['goal'].keys() if domain not in ['restaurant','message','topic']]) == 0: 
                yield {'id': did,
                       'goal': dial['goal'],
                       'turns_iterator': self._make_turn_iter(dial['log'])}

    def _convert_log(self, das):
            for slot_name, slot_value in das['restaurant']['semi'].items():
                if len(slot_value) > 0:
                    yield {'act': 'inform',
                           'slots': (('{}-{}'.format('restaurant', slot_name), slot_value),)}


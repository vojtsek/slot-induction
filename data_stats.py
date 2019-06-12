import argparse

from readers import CamRestReader, MultiWOZReader, MultiWOZRestaurantReader, dact
from kb import KnowledgeBase

CAM = 'camrest'
WOZ = 'multiwoz'
WOZREST = 'multiwoz-rest'

def make_query_from_dacts(dacts):
    return [{dact.slot: dact.value for dact in dacts}]


def accumulate_utts(utts, turn):
    utts.append(turn['usr_trn'])
    utts.append(turn['sys_trn'])
    return utts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=CAM)
    parser.add_argument('--db', default=None)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    reader_factory = None
    if args.dataset == CAM:
        reader_factory = CamRestReader
    elif args.dataset == WOZ:
        reader_factory = MultiWOZReader
    elif args.dataset == WOZREST:
        reader_factory = MultiWOZRestaurantReader

    if reader_factory is not None:
        reader = reader_factory(args.data_path)
    else:
        return

    kb = None
    if args.db is not None:
        kb = KnowledgeBase()
        schema = {'address', 'area', 'food', 'location',
                  'phone', 'pricerange', 'postcode',
                  'type', 'id', 'name'}
        kb.load_from_file(args.db, 'Restaurants', schema=schema)
        table = kb.table('Restaurants')
    for dial in reader.dial_iter():
        print(dial['id'])

#    for utt in reader.call_for_every_turn(accumulate_utts, []):
 #       print(utt)

#    for turn in reader.turns_iter():
#        if kb is not None:
#            print(len(kb.select(table, make_query_from_dacts(turn['usr_slu']))))


if __name__ == '__main__':
    main()

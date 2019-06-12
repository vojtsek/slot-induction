import tinydb
import json
import random
import logging

logger = logging.getLogger()

class KnowledgeBase:

    def __init__(self, db_file='../data/db.json'):
        logger.info('Using db file %s', db_file)
        self.db = tinydb.TinyDB(db_file)

    def table(self, table_name):
        return self.db.table(table_name)

    def tables(self):
        return self.db.tables()

    def _check_schema(self, entry, schema):
        """
        Checks if the entry conforms given schema
        :param entry: dict to be checked
        :param schema: set of fields that must be contained in the entry
        :return: True if supplied entry conforms the schema
        """
        if schema is None:
            return True
        return all([field in entry for field in schema])


    def load_from_file(self, json_file, table_name, force_insert=False, schema=None):
        """
        loads data from json into the db
        :param json_file: path to the json file with data. It's supposed to contain list of dicts
        :param table_name: name of the table the data should be stored in
        :param force_insert: if True, inserts to alredy existing table
        :param schema: dict; if supplied all the loaded entries must conform the provided schema
        :return: reference for the table
        """
        logger.info('Loading data from %s into table "%s"', json_file, table_name)
        if table_name in self.tables() and not force_insert:
            logger.warning('Table %s already exists, use force_insert=True to append the data', table_name)
            return self.table(table_name)
        table = self.table(table_name)
        with open(json_file, 'r') as f:
            db_content = json.load(f)
            for n, entry in enumerate(db_content):
                if self._check_schema(entry, schema):
                    table.insert(entry)
                else:
                    logger.warning('Entry "%s" does not conform the schema!', entry)
            logger.info('Inserted %d records into %s', n, table_name)
        return table


    def select(self, table, query_dnf):
        """
        just wraps a symbolic query to db_backend-specific query
        :param table: table to be searched
        :param query_dnf: formula in DNF [{a: 1, b:2}, {b: 3}] <=> (a==1 & b==2) | (b==3)
        :return: result of the query
        """
        if len(query_dnf[0]) == 0:
            return QueryResult(None)
        query_str = []
        for conjunction in query_dnf:
            conj = []
            for field, val in conjunction.items():
                conj.append('(tinydb.where(\'{}\') == \'{}\')'.format(field, val))
            query_str.append('({})'.format(' & '.join(conj)))
        query_str = '({})'.format(' | '.join(query_str))
        return QueryResult(table.search(eval(query_str)))


class QueryResult:

    def __init__(self, res=None):
        self._result = res if res is not None else []
        self.empty = len(self._result) == 0

    def __len__(self):
        return len(self._result)

    @property
    def result(self):
        return self._result

    @property
    def random_result(self):
        return random.choice(self._result)

    def distinct_values_for_slot(self, slot):
        return set([res[slot] for res in self._result])

    def only_fields(self, fields):
        results = []
        for res in self._result:
            results.append({x:y for x, y in res.items() if x in fields})
        return results


if __name__ == '__main__':
    kb = KnowledgeBase()
    schema = {'address', 'area', 'food', 'location',
              'phone', 'pricerange', 'postcode',
              'type', 'id', 'name'}
    kb.load_from_file('../data/CamRestDb.json', 'Restaurants', schema=schema)
    table = kb.table('Restaurants')
    result = kb.select(table, [{'area': 'centre', 'food': 'italian'}, {'area': 'north'}])
    print(result.distinct_values_for_slot('food'))

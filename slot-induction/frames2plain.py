import sys
import json

with open(sys.argv[1], 'rt') as f:
    for line in f:
        line = line.split()
        name = line[0]
        content = ' '.join(line[4:])
        content = json.loads(content)
        print(' '.join(content))

import sys
import json

with open(sys.argv[1], 'rt') as f:
    for line in f:
        line = line.split()
        name = line[0]
        try:
            if int(line[5]) == 0:
                print('dummy')
                continue
        except:
            pass
        content = ' '.join(line[5:])
        content = json.loads(content)
        print(' '.join(content))

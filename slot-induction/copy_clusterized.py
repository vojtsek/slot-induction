import sys

last = None
f = None
infile = sys.argv[1]
outfile = sys.argv[2]
for line in sys.stdin:
    line = line.split()
    d = line[0]
    if d != last:
        if f is not None:
            f.close()
        f = open(d + '/' + infile, 'rt')
    utt = f.readline().strip()
    last = d
    with open(d + '/' + outfile + line[2], 'a') as cf:
        print(utt, file=cf)

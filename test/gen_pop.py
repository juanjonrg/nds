#!/usr/bin/env python3

import sys, datetime, random

n = int(sys.argv[1])
m = int(sys.argv[2])

def gen_pop(n, m, filename):
    with open(filename, 'w') as f:
        for i in range(n):
            f.write(' '.join(map('{:.2f}'.format, [random.random() for j in range(m)])))
            f.write('\n')

date = datetime.datetime.today().strftime('%Y%m%d_%H%M%S.%f')[:-3]
filename = 'pops/rand_{}_{}_{}.txt'.format(n, m, date)
gen_pop(n, m, filename)
print("Population generated:", filename)

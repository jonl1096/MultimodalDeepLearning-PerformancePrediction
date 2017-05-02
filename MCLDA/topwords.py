import sys
from operator import itemgetter

if len(sys.argv) != 2:
	print 'usage: python topwords.py <phi_file>'
	exit()

phi = []

with open(sys.argv[1], 'r') as f:
	Z = len(f.readline().split()) - 1  # number of topics
	for z in range(0, Z): phi.append({})
	f.seek(0)

	for line in f:
		tokens = line.split()
		word = tokens.pop(0)
		for z in range(0, Z):
			phi[z][word] = float(tokens[z])

for z in range(0, Z):
	print 'Topic', z
	words = sorted(phi[z].items(), key=itemgetter(1), reverse=True)
	w = 0
	for word, p in words:
		print word, p
		w += 1
		if w == 20: break
	print ''

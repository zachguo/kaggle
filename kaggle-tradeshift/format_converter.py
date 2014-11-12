#!/usr/bin/env python

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
LABEL_PATH = 'data/trainLabels.csv'

# prepare feature hashing for libsvm, no need for vw.
COLS_HASH = [1, 2, 3, 4, 10, 11, 12, 13, 14, 24, 25, 26, 30, 31, 32, 33, 34, 35, 
41, 42, 43, 44, 45, 55, 56, 57, 61, 62, 63, 64, 65, 71, 72, 73, 74, 75, 
85, 86, 87, 91, 92, 93, 94, 95, 101, 102, 103, 104, 105, 115, 116, 117, 126, 
127, 128, 129, 130, 140, 141, 142]
COLS_HASH = [1 if i in COLS_HASH else 0 for i in xrange(1, 146)]
D = 2 ** 20

# generator for labels
I_LABEL = 33
def label_gen():
	with open(LABEL_PATH, 'r') as fin_labels:
		fin_labels.readline()
		for line in fin_labels:
			yield line.strip().split(',')[I_LABEL]
LABELS = label_gen()

def libsvm(filepath, mode):
	with open(filepath.rpartition('.')[0]+'.libsvm', 'w') as fout, open(filepath, 'r') as fin:
		fin.readline() # skip header line
		for line in fin:
			row = line.split(',')
			row_new = [next(LABELS)] if mode == 'train' else ['0']
			for i in xrange(1, 146):
				if COLS_HASH[i-1]:
					hashed_val = abs(hash(str(i) + '_' + row[i])) % D
					# prepend 20 to avoid conficting with existing numeric vars
					row_new.append('20{}:{}'.format(hashed_val, 1))
				else:
					row_new.append('{}:{}'.format(i, row[i]))
			fout.write(' '.join(row_new))

def vw(filepath, mode):
	with open(filepath.rpartition('.')[0]+'.vw', 'w') as fout, open(filepath, 'r') as fin:
		fin.readline() # skip header line
		for line in fin:
			row = line.strip().split(',')
			if mode == 'train':
				label = next(LABELS)
				row_new = ['1' if label else '-1']
			else:
				row_new = ['1']
			row_new.append('1') # add weight
			ns_n = [row[0] + '|n'] # numeric namespace
			ns_q = ['|q'] # quadratic namespace
			ns_h = ['|h'] # hashed namespace
			for i in xrange(1, 146):
				if COLS_HASH[i-1]:
					hashed_val = str(abs(hash(str(i) + '_' + row[i])) % D)
					if i in [4,35,65,95]:
						ns_q.append(hashed_val)
					else:
						ns_h.append(hashed_val)
				else:
					ns_n.append('{}:{}'.format(i, row[i]))
			fout.write(' '.join(row_new + ns_n + ns_q + ns_h) + '\n')

if __name__ == '__main__':
	from sys import argv
	if argv[1] == 'libsvm':
		libsvm(TRAIN_PATH, 'train')
		libsvm(TEST_PATH, 'test')
	elif argv[1] == 'vw':
		vw(TRAIN_PATH, 'train')
		vw(TEST_PATH, 'test')
	else:
		print 'Wrong format'
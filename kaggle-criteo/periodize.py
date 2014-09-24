#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys, random

def determine_period(input_file):
	"""input: ctr_ma_old.csv"""
	csvr = csv.reader(open(input_file, 'rb'), delimiter=',')
	ctr = [float(row[1]) for row in csvr]
	gap = 100000
	i = 0
	while i < (len(ctr) - gap):
		if ctr[i+gap] - ctr[i] > 0.02:
			print i
			i += gap
		else:
			i += 1
# find split indice:
# [6467204, 12562341, 19321785, 26289821, 33076258, 39700372]

# num of rows for each period:
# 6467204
# 6095137
# 6759444
# 6968036
# 6786437
# 6624114
# 6140245
# num of rows of test:
# 6042135

def periodize():
	"""
	Convert each day into equal length period by randomly deletion.
	Return the a list of deleted IDs.
	"""
	deleted = []
	split_indice = [6467204, 12562341, 19321785, 26289821, 33076258, 39700372]
	length = 6042135
	split_indice = [0] + split_indice + [45840617]
	for i in range(len(split_indice)-1):
		start = split_indice[i]
		end = split_indice[i+1]
		deleted += random.sample(range(start,end), end-start-length)
	return deleted

def train_periodized(deleted):
	deleted = set(deleted)
	i = 0
	with open('data/train.csv', 'rb') as fin:
		with open('data/train_periodized.csv', 'wb') as fout:
			for line in fin:
				if i not in deleted:
					fout.write(line)
				i += 1

def add_bgctr(input_file, input_file_bgctr, output_file):
	with open(input_file, 'rb') as fin_csv:
		with open(input_file_bgctr, 'rb') as fin_bgctr:
			with open(output_file, 'wb') as fout:
				fout.write(fin_csv.readline().strip()+',bgctr\n')
				for line in fin_csv:
					fout.write(line.strip()+','+fin_bgctr.readline())

if __name__ == '__main__':
	input_file = sys.argv[1]
	# determine_period(input_file)
	# train_periodized(periodize())
	add_bgctr('data/train_periodized.csv', 'data/train_bgctr.txt', 'data/train_with_bgctr.csv')
	add_bgctr('data/test.csv', 'data/test_bgctr.txt', 'data/test_with_bgctr.csv')

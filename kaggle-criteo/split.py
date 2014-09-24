#!/usr/bin/env python
# -*- coding: utf-8 -*-

# partially preprocess and split data
# 1. remove categorical vars that contain too many classes
# 2. log transform I* vars
# 3. split data into 3 parts accroding to C20

import csv
from collections import defaultdict
from math import log

DATAPATH = 'data/'

# make sure that C20 has same classes in train and test, print out classes
def check_C20(input_file):
	csvr = csv.DictReader(open(input_file, 'rb'), delimiter=',')
	classes = defaultdict(int)
	for row in csvr:
		classes[row['C20']] += 1
		# if len(classes) == 4: break # early stop
	return classes

cols_tolog = ['I{}'.format(i) for i in range(1,14)]
cols_tohash = ['C6', 'C9', 'C14', 'C17', 'C22', 'C23']

def cleanup_row(row, mode):
	row_cleaned = [row['Id']]
	if mode == 'train':
		row_cleaned.append(row['Label'])
	row_cleaned.append(row['bgctr'])
	for col in cols_tolog:
		if row[col]:
			val = float(int(row[col]))
			if val < 0: 
				val = 0.
			val = log((1.+val))
			row_cleaned.append(str(val))
		else:
			row_cleaned.append('')
	for col in cols_tohash:
		if row[col]:
			row_cleaned.append(str(int(row[col], 16)))
		else:
			row_cleaned.append('')
	return ','.join(row_cleaned)+'\n'

def split(input_file, mode, by):

	if mode == 'train':
		cols = ['Id', 'Label', 'bgctr'] + cols_tolog + cols_tohash
	elif mode == 'test':
		cols = ['Id', 'bgctr'] + cols_tolog + cols_tohash
	else:
		print 'Invalid mode'
		return

	csvr = csv.DictReader(open(input_file, 'rb'), delimiter=',')
	if by == 'C20':
		header = ','.join(cols)+'\n'
		fout_ = open(DATAPATH+'submodels_by_{}/{}_.csv'.format(by, mode), 'wb')
		fout_b1252a9d = open(DATAPATH+'submodels_by_{}/{}_b1252a9d.csv'.format(by, mode), 'wb')
		fout_5840adea = open(DATAPATH+'submodels_by_{}/{}_5840adea.csv'.format(by, mode), 'wb')
		fout_a458ea53 = open(DATAPATH+'submodels_by_{}/{}_a458ea53.csv'.format(by, mode), 'wb')
		fout_.write(header)
		fout_b1252a9d.write(header)
		fout_5840adea.write(header)
		fout_a458ea53.write(header)
		for row in csvr:
			c20 = row['C20']
			if c20 == '':
				fout_.write(cleanup_row(row, mode))
			elif c20 == 'b1252a9d':
				fout_b1252a9d.write(cleanup_row(row, mode))
			elif c20 == '5840adea':
				fout_5840adea.write(cleanup_row(row, mode))
			elif c20 == 'a458ea53':
				fout_a458ea53.write(cleanup_row(row, mode))
		fout_.close()
		fout_b1252a9d.close()
		fout_5840adea.close()
		fout_a458ea53.close()

	elif by == 'days':
		cols += ['C20']
		header = ','.join(cols)+'\n'
		counter = 0
		fout = open('whatever.txt', 'wb')
		for row in csvr:
			if counter % 6042135 == 0:
				fout.close()
				fout = open(DATAPATH+'submodels_by_{}/{}_day{}.csv'.format(by, mode, counter/6042135 + 1), 'wb')
				fout.write(header)
			counter += 1
			fout.write(cleanup_row(row, mode))
		fout.close()

if __name__ == '__main__':
	# print 'test:', check_C20(DATAPATH+"test_with_bgctr.csv")
	# test: set(['', 'b1252a9d', '5840adea', 'a458ea53']) 
	# {'': 2664224, 'b1252a9d': 1117304, '5840adea': 1125786, 'a458ea53': 1134821}
	# print 'train:', check_C20(DATAPATH+"train_with_bgctr.csv")
	# train: set(['b1252a9d', '', '5840adea', 'a458ea53'])
	
	# # split whole data by C20
	# split(DATAPATH+'test_with_bgctr.csv', 'test', 'C20')
	# split(DATAPATH+'train_with_bgctr.csv', 'train', 'C20')

	# split whole data by days
	# split(DATAPATH+'test_with_bgctr.csv', 'test', 'days')
	# split(DATAPATH+'train_with_bgctr.csv', 'train', 'days')

	# ensemble cv
	# DATAPATH = 'data/cv_ensemble/'
	# split(DATAPATH+'xab', 'test', 'C20')
	# split(DATAPATH+'xaa', 'train', 'C20')
	pass
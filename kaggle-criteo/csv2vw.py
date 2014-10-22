#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys
from math import log

def featurify(row, feature_names):

	features = []
	try:
		label = row['Label']
		if int(label) == 0: label = -1 # required by logistic/hinge loss functions
	except: # for test data
		label = 1
	features.append("{} 1 {}|n".format(label, row['Id']))
	
	# for namespace in ['n', 'b', 'c', 's']: # put numeric vars first
	for namespace in ['n', 'c', 's']: # put numeric vars first
		if namespace == 'n':
			for c in feature_names[namespace]:
				if row[c]: # skip missing values in num vars
					# val = float(int(row[c]))
					val = float(row[c])
					if val < 0: 
						val = 0.
					# log transformation of I* vars
					val = log((1.+val))
					features.append("{}:{}".format(c, val))
		# elif namespace == 'b':
		# 	for c in feature_names[namespace]:
		# 		if row[c]: # skip missing values in num vars
		# 			features.append("{}:{}".format(c, row[c]))
		else:
			features.append("|%s" % namespace[0])
			for c in feature_names[namespace]:
				features.append(row[c])

	features = " ".join(features)
	features += "\n"
	return features

if __name__ == '__main__':
	input_file, output_file = sys.argv[1:3]
	csvr = csv.DictReader(open(input_file, 'rb'), delimiter=',')
	feature_names = list(set(csvr.fieldnames) - set(['Label', 'Id']))
	num_feature_names = [x for x in feature_names if x.startswith('I')]
	cat_feature_names = [x for x in feature_names if x.startswith('C')]
	small_cat_feature_names = ["C2", "C5", "C6", "C8", "C9", "C14", "C17", "C20", "C22", "C23", "C25"]
	large_cat_feature_names = list(set(cat_feature_names) - set(small_cat_feature_names))
	feature_names = {'n':num_feature_names, 'c':large_cat_feature_names, 's':small_cat_feature_names}#, 'b':['bgctr']}
	with open(output_file, 'wb') as fout:
		for row in csvr:
			fout.write(featurify(row, feature_names))

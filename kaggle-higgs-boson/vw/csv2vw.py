#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys

def featurify(row, feature_names):

	features = []
	try:
		label = row['Label']
		if label == 'b': label = 0
		elif label == 's': label = 1
	except KeyError:
		label = '1'

	features.append("{} 1 {}|n".format(label, row['EventId']))
	for c in feature_names:
		features.append("{}:{}".format(c, row[c]))

	features = " ".join(features)
	features += "\n"
	return features

if __name__ == '__main__':
	input_file, output_file = sys.argv[1:3]
	csvr = csv.DictReader(open(input_file, 'rb'), delimiter=',')
	feature_names = list(set(csvr.fieldnames) - set(['999', 'Label', 'Weight', 'EventId']))
	with open(output_file, 'wb') as fout:
		for row in csvr:
			fout.write(featurify(row, feature_names))

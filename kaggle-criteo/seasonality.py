#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys

if __name__ == '__main__':
	input_file, output_file = sys.argv[1:3]
	csvr = csv.DictReader(open(input_file, 'rb'), delimiter=',')
	ctr = 0.22
	decay = 0.9999
	with open(output_file, 'wb') as fout:
		for row in csvr:
			ctr = decay * ctr + (1.0 - decay) * float(row['Label'])
			fout.write("{},{}\n".format(row['Id'], ctr))

##!/usr/bin/env python

from sys import argv
import math

def single_var_xgb(filename_sub, filename_var, index_var):
	# blend submission with xgboost result of a single single_var
	filename_out = filename_sub.rpartition('.')[0] + '_xgblended' + index_var + '.csv'
	with open(filename_sub, 'r') as fin_sub, open(filename_var, 'r') as fin_var, open(filename_out, 'w') as fout:
		fout.write(fin_sub.readline()) # header
		for line in fin_sub:
			id_label, val_sub = line.strip().split(',')
			if id_label.endswith(index_var):
				val_var = fin_var.readline().strip()
				val_new = (float(val_sub) + float(val_var)) / 2.
				fout.write('{},{}\n'.format(id_label, val_new))
			else:
				fout.write(line)

def single_var_vw(filename_sub, filename_var, index_var):
	filename_out = filename_sub.rpartition('.')[0] + '_vwblended' + index_var + '.csv'
	with open(filename_sub, 'r') as fin_sub, open(filename_var, 'r') as fin_var, open(filename_out, 'w') as fout:
		fout.write(fin_sub.readline()) # header
		for line in fin_sub:
			id_label, val_sub = line.strip().split(',')
			if id_label.endswith(index_var):
				val_var = float(fin_var.readline().strip().split()[0])
				val_var = 1 / (1 + math.exp(-val_var))
				val_new = (float(val_sub) + float(val_var)) / 2.
				fout.write('{},{}\n'.format(id_label, val_new))
			else:
				fout.write(line)

def main():
	if argv[1] == '--xgbvar':
		single_var_xgb(argv[2], argv[3], argv[4])
	elif argv[1] == '--vwvar':
		single_var_vw(argv[2], argv[3], argv[4])

if __name__ == '__main__':
	main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from csv import QUOTE_NONNUMERIC

train = pd.read_csv('avito_train.tsv', sep='\t')
test = pd.read_csv('avito_test.tsv', sep='\t')

subcats_names =  train['subcategory'].unique()

for i in range(len(subcats_names)):
	scn = subcats_names[i]
	train[train['subcategory'] == scn].to_csv('train_%d.tsv' % i, sep='\t', quoting=QUOTE_NONNUMERIC, index=False)
	test[test['subcategory'] == scn].to_csv('test_%d.tsv' % i, sep='\t', quoting=QUOTE_NONNUMERIC, index=False)

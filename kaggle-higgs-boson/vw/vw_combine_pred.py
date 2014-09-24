#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

submodel1 = pd.read_csv('data/predictions.txt', header=None, names=['val','EventId'], sep=' ', index_col='EventId').sort_index()

combined = (submodel1).div(1.0)
combined['EventId'] = combined.index
combined.sort('val', ascending=True, inplace=True)
combined['RankOrder'] = range(1, combined.shape[0]+1)
i_split_b_s = combined.shape[0]*85/100
combined['Class'] = i_split_b_s * ['b'] + (combined.shape[0] - i_split_b_s) * ['s']
combined = combined[['EventId', 'RankOrder', 'Class']]
combined.to_csv('data/predictions_combined_for_kaggle.csv', sep=',', header=True, index=False)
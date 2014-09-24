#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

submodel1 = pd.read_csv('predictions_submodel1.txt', header=None, names=['val','id'], sep=' ', index_col='id').sort_index()
submodel2 = pd.read_csv('predictions_submodel2.txt', header=None, names=['val','id'], sep=' ', index_col='id').sort_index()
submodel3 = pd.read_csv('predictions_submodel3.txt', header=None, names=['val','id'], sep=' ', index_col='id').sort_index()
submodeltrigram = pd.read_csv('predictions_submodeltrigram.txt', header=None, names=['val','id'], sep=' ', index_col='id').sort_index()
onemodelbigram = pd.read_csv('predictions_onemodelbigram.txt', header=None, names=['val','id'], sep=' ', index_col='id').sort_index()

combined = (submodel1 + submodel2 + submodel3 + submodeltrigram.multiply(2) + onemodelbigram).div(6.0)
combined['id'] = combined.index
combined.sort('val', ascending=0, inplace=True)
combined[['id']].to_csv('predictions_combined_for_kaggle.txt', sep=' ', header=['Id'], index=False)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from numpy import isfinite, log
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

print "load data..."
train = pd.read_csv("data/train.csv", index_col='EventId')
test = pd.read_csv("data/test.csv", index_col='EventId')
cols = list(set(train.columns) - set(['Label', 'Weight']))

# remove outliers: PRI_met & DER_pt_h & DER_pt_tot > 2500, a single data row
train = train[train['PRI_met'] <= 2500]
# check missing values other than -999, fortunately none
assert all(train[cols+['Weight']].applymap(isfinite).apply(all, axis=0))
assert all(test[cols].applymap(isfinite).apply(all, axis=0))
# find all possible -999 patterns in training and testing data
train['999'] = train[cols].applymap(lambda x:str(int(x == -999))).sum(axis=1).apply(str)
test['999'] = test[cols].applymap(lambda x:str(int(x == -999))).sum(axis=1).apply(str)
# train['999'] = train['999'] + train['PRI_jet_num'].apply(str)
# test['999'] = test['999'] + test['PRI_jet_num'].apply(str)
nnnpatterns = sorted(train['999'].unique())
# luckily train and test data share the same -999 patterns
assert nnnpatterns == sorted(test['999'].unique())
# map pattern to an int label
nnnp_map = dict(zip(nnnpatterns, range(len(nnnpatterns))))
train['999'] = train['999'].map(nnnp_map)
test['999'] = test['999'].map(nnnp_map)
# # normalize some non-negative numeric features, THIS HARM PREDICTION DRASTICALLY
# cols_log = ['DER_pt_h', 'PRI_tau_pt', 'DER_mass_jet_jet', 
# 			'DER_deltar_tau_lep', 'PRI_jet_all_pt', 'DER_mass_MMC',
# 			'PRI_met', 'DER_pt_tot', 'DER_mass_vis',
# 			'PRI_jet_leading_pt', 'DER_sum_pt', 'PRI_jet_subleading_pt',
# 			'PRI_met_sumet', 'PRI_lep_pt', 'DER_mass_transverse_met_lep', 
# 			'DER_pt_ratio_lep_tau']
# train[cols_log] = train[cols_log].applymap(lambda x: log(x+0.000001) if x!= -999 else -999)

def fit_predict(clf, train, test):
	print clf
	clf.fit(train[cols], train['Label'].values)
	print clf.score(train[cols], train['Label'].values)
	return clf.predict_proba(test[cols])[:,1]

def prob2rank(prob):
	return [item[1] for item in sorted(zip(sorted(zip(prob, range(len(prob)))), range(1, len(prob)+1)), key=lambda x:x[0][1])]

def ensembled_fit_predict(train, test):
	pred = pd.Series([0]*test.shape[0], index=test.index)
	pred = pred.add(pd.Series(fit_predict(GradientBoostingClassifier(n_estimators=100, max_depth=7, min_samples_leaf=100, max_features='log2'), train, test), index=test.index))
	pred = pred.add(pd.Series(fit_predict(RandomForestClassifier(n_estimators=100, criterion='entropy'), train, test), index=test.index))
	return pred

def singlemodel(train, test, cols):
	print "run single model..."
	# init output df
	output_df = pd.Series(index=[], name='prob')
	# single model
	nnn2nan = lambda x: "NaN" if x == -999 else x
	train = train.applymap(nnn2nan)
	test = test.applymap(nnn2nan)
	imputer = Imputer()
	scaler = MinMaxScaler()
	train[cols] = imputer.fit_transform(train[cols])
	test[cols] = imputer.transform(test[cols])
	train[cols] = scaler.fit_transform(train[cols])
	test[cols] = scaler.transform(test[cols])
	output_df = output_df.append(ensembled_fit_predict(train, test))
	output_df = prob2rank(output_df.values)
	return output_df

def submodels(train, test, cols):
	print "run submodels..."
	# init output df
	output_df = pd.Series(index=[], name='prob')
	# run submodels
	for name, df_train in train.groupby('999'):
		# feature scaling
		scaler = MinMaxScaler()
		df_train[cols] = scaler.fit_transform(df_train[cols])
		df_train.to_csv('data/train_%d.csv' % name)
		# corresponding testing data
		df_test = test[test['999'] == int(name)]
		df_test[cols] = scaler.transform(df_test[cols])
		df_test.to_csv('data/test_%d.csv' % name)
		# fit and predict
		print 'Submodel', nnnpatterns[int(name)]
		output_df = output_df.append(ensembled_fit_predict(df_train, df_test))
	idx = output_df.index
	output_df = pd.Series(prob2rank(output_df.values), index=idx, name='prob')
	return output_df

def combine_output(output_df, xgboost_df):
	print output_df
	print "combine with xgboost result..."
	output_df.sort_index(inplace=True)
	xgboost_df.sort_index(inplace=True)
	output_df['prob'] = xgboost_df['RankOrder'] + output_df['prob']
	print output_df
	output_df.sort('prob', ascending=True, inplace=True)
	output_df['RankOrder'] = range(1, output_df.shape[0]+1)
	for signal_prop in [15,16]:# 15% as signal
		i_split_b_s = output_df.shape[0]*(100 - signal_prop)/100
		output_df['Class'] = i_split_b_s * ['b'] + (output_df.shape[0] - i_split_b_s) * ['s']
		output_df = output_df[['RankOrder', 'Class']]
		output_df.to_csv('data/submission%d.csv' % signal_prop, sep=',', header=True, index=True)

def main():
	output_df = pd.DataFrame(submodels(train, test, cols))
	xgboost_df = pd.read_csv('data/xgboost_df.csv', index_col='EventId')
	combine_output(output_df, xgboost_df)

if __name__ == '__main__':
	main()
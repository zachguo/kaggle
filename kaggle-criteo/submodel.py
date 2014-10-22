#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

nrows_dict = {'C20': 
				{'':(18607630, 2664225), 
				'5840adea':(7873220, 1125787), 
				'a458ea53':(7781656, 1134822), 
				'b1252a9d':(8032443, 1117305)},
			  'days':
				{'day1':(6042136, 6042136),
				'day2':(6042136, 6042136),
				'day3':(6042136, 6042136),
				'day4':(6042136, 6042136),
				'day5':(6042136, 6042136),
				'day6':(6042136, 6042136),
				'day7':(6042136, 6042136)}}

def dummify(data, cols_cat):
	# dummify categotical vars
	for col in cols_cat:
	    print '  Dummified', col
	    data = pd.concat([data, pd.get_dummies(data[col], dummy_na=True, prefix=col)], axis=1)
	    data.drop(col, inplace=True, axis=1)
	return data

def le(series, labels):
	codes = range(1, len(labels)+1)
	lcmap = dict(zip(labels, codes))
	return series.map(lcmap).fillna(0)

def encode_label(cols_cat, train, test=None):
	# encode_label categotical vars, for tree models
	for col in cols_cat:
		print '  Encoding', col
		if test is not None:
			labels = list(set(train[col].unique()) | set(test[col].unique()))
			train[col] = le(train[col], labels).values
			test[col] = le(test[col], labels).values
		else:
			labels = train[col].unique()
			train[col] = le(train[col], labels).values
	if test is not None:
		return train, test
	else:
		return train

def load_n_clean_data(tag, by, cv, load=True, output=False):
	print "Load data..."
	if load:
		train = pd.read_csv('data/submodels_by_{}/train_{}_cleaned.csv'.format(by, tag), index_col='Id', nrows=nrows_dict[by][tag][0])
	else:
		train = pd.read_csv('data/submodels_by_{}/train_{}.csv'.format(by, tag), index_col='Id', nrows=nrows_dict[by][tag][0])

		cols_all = train.columns
		# cols_num = [x for x in cols_all if x.startswith('I')] + ['bgctr']
		cols_cat = [x for x in cols_all if x.startswith('C')]

		# encode categorical labels to int
		train[cols_cat] = train[cols_cat].astype(str)
		if cv:
			train = encode_label(cols_cat, train)
		else:
			if by == 'C20':
				test = pd.read_csv('data/submodels_by_{}/test_{}.csv'.format(by, tag), index_col='Id', nrows=nrows_dict[by][tag][1])
			elif by == 'days':
				test = pd.read_csv('data/submodels_by_{}/test_day1.csv'.format(by), index_col='Id', nrows=nrows_dict[by][tag][1])
			test[cols_cat] = test[cols_cat].astype(str)
			train, test = encode_label(cols_cat, train, test)
		
		if output:
			train.to_csv('data/submodels_by_{}/train_{}_cleaned.csv'.format(by, tag))
			if not cv:
				if by == 'C20':
					test.to_csv('data/submodels_by_{}/test_{}_cleaned.csv'.format(by, tag))
				elif by == 'days':
					test.to_csv('data/submodels_by_{}/test_day1_cleaned.csv'.format(by))

	# prepare for sklearn classifier
	cols = list(set(train.columns)-set(['Label']))
	X = train[cols].fillna(0.)
	y = train['Label'].values
	del train
	if cv:
		return X, y
	else:
		test = test[cols].fillna(0.)
		return X, y, test

def split_train_test(X, y):
	cutpoint = len(y)*6/7
	X_train = X[:cutpoint]
	X_test = X[cutpoint:]
	y_train = y[:cutpoint]
	y_test = y[cutpoint:]
	return X_train, X_test, y_train, y_test

def cv_submodel(tag, by, clf, clf_str, params):
	X, y = load_n_clean_data(tag, by, cv=True)
	X_train, X_test, y_train, y_test = split_train_test(X, y)
	print "Grid Search CV ..."
	gscv = GridSearchCV(clf, 
		params, 
		scoring=make_scorer(log_loss, needs_proba=True, greater_is_better=False), 
		verbose=4)
	gscv.fit(X_train, y_train).score(X_test, y_test)
	print "Best params:", gscv.best_params_
	print "Generate Log-Loss ..."
	clf = eval(clf_str.format(','.join(["{}={}".format(k,v) for k,v in gscv.best_params_.items()])))
	clf.fit(X_train, y_train)
	print log_loss(y_test, clf.predict_proba(X_test))
	print clf.feature_importances_

def real_submodel(tag, by):
	print "Classify submodel_by_{}_{} ...".format(by, tag)
	X, y, X_test = load_n_clean_data(tag, by, load=False, cv=False)
	print "Build model ..."
	# clf = AdaBoostClassifier(ExtraTreesClassifier(n_jobs=-1, 
	# 	n_estimators=100, 
	# 	min_samples_leaf=9, 
	# 	max_depth=20, 
	# 	verbose=4), n_estimators=10)
	clf = ExtraTreesClassifier(n_jobs=-1, 
		n_estimators=200, 
		min_samples_leaf=9, 
		max_depth=30, 
		verbose=4)
	clf.fit(X, y)
	pred = clf.predict_proba(X_test)
	print pred

	print "Dump precious stuff in case of crash ..."
	import pickle
	# with open('output/submodels_by_{}/pred_{}.cache'.format(by, tag), 'w') as fout_pred:
	# 	pickle.dump(pred, fout_pred)
	# clf occupy too much space of disk
	# with open('output/submodels_by_{}/clf_{}.cache'.format(by, tag), 'w') as fout_clf:
	# 	pickle.dump(clf, fout_clf)

	label1_idx = clf.classes_.tolist().index(1)
	X_test['Predicted'] = [item[label1_idx] for item in pred]
	return X_test['Predicted']

def real_submodels_by_C20():
	output = real_submodel('', 'C20')
	for tag in ['5840adea', 'a458ea53', 'b1252a9d']:
		output = output.append(real_submodel(tag, 'C20'))
	output.to_csv('output/submission_submodels_by_C20.csv', header=True)

def real_submodels_by_days():
	for tag in nrows_dict['days']:
		output = real_submodel(tag, 'days')
		output.to_csv('output/submission_submodels_by_days_{}.csv'.format(tag), header=True)

def gscv_submodels():
	params = {'n_estimators':[200], 
			  'max_depth':[20], 
			  'min_samples_leaf':[9], 
			  'criterion':['gini', 'entropy'],
			  # 'min_samples_split':[50, 100]
			  }
	cv_submodel('5840adea', 'C20', ExtraTreesClassifier(n_jobs=-1), 'ExtraTreesClassifier(n_jobs=-1,{})', params)

if __name__ == '__main__':
	real_submodels_by_C20()
	# real_submodels_by_days()

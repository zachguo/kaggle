#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

REAL_PATH = "output/"
CV_PATH = "output/cv_ensemble/"

def readcsv(tag, mode):
	if mode == 'cv':
		filename = CV_PATH+"submission_%s.csv" % tag
	elif mode == 'real':
		filename = REAL_PATH+"submission_%s.csv" % tag
	df = pd.read_csv(filename, sep=',', header=0, index_col='Id')
	df.columns = [tag]
	return df

def load_all_submissions(tags, mode):
	base = readcsv(tags[0], mode)
	for tag in tags[1:]:
		base = add_col(base, readcsv(tag, mode))
	if mode == 'cv': 
		base = add_col(base, readcsv('goldstandard', 'cv'))
	return base

def add_col(base, df_tag):
	return pd.concat([base, df_tag], axis=1)

def averaging(df, output=None):
	cols = list(set(df.columns) - set(['goldstandard']))
	df = df[cols].sum(axis=1).div(len(cols))
	if output:
		df.name = 'Predicted'
		df.columns = ['Predicted']
		df.to_csv(output, header=True)
	return df.values

def get_X_y(df):
	cols = list(set(df.columns) - set(['goldstandard']))
	X = df[cols]
	y = df['goldstandard'].values
	return X, y

def cv_clf(X, y, clf, params):
	print "Grid Search CV ..."
	gscv = GridSearchCV(clf, 
		params, 
		scoring=make_scorer(log_loss, needs_proba=True, greater_is_better=False), 
		verbose=4)
	gscv.fit(X, y)
	print "Best params:", gscv.best_params_

def cv_avg(df):
	print df.shape
	print "CV for averaging ensemble:"
	X, y = get_X_y(df)
	for train_index, test_index in KFold(df.shape[0], 7):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print log_loss(y_test, averaging(X_test))

def cv_tree(df):
	print df.shape
	print "CV for tree ensemble:"
	params = {'n_estimators':[100], 
			  'max_depth':[30], 
			  'min_samples_leaf':[9], 
			  'criterion':['entropy'],
			  }
	X, y = get_X_y(df)
	cv_clf(X, y, ExtraTreesClassifier(n_jobs=-1), params)

def cv_lr(df):
	print df.shape
	print "CV for lr ensemble:"
	params = {'C':[1.]}
	X, y = get_X_y(df)
	cv_clf(X, y, LogisticRegression(), params)

def real(tags, clf):
	df_train = load_all_submissions(tags, 'cv')
	X_test = load_all_submissions(tags, 'real')
	X, y = get_X_y(df_train)
	clf.fit(X, y)
	pred = clf.predict_proba(X_test)
	label1_idx = clf.classes_.tolist().index(1)
	X_test['Predicted'] = [item[label1_idx] for item in pred]
	return X_test['Predicted']

if __name__ == '__main__':
	args = sys.argv[1:]

	# CV
	# df = load_all_submissions(args, 'cv')
	# cv_avg(df)
	# # # cv_tree(df)
	# cv_lr(df)

	# REAL
	# averaging(load_all_submissions(args, 'real'), 'output/submission_avg_{}.csv'.format(''.join(args))) # avg
	# clf = ExtraTreesClassifier(n_jobs=-1, criterion='entropy', min_samples_leaf=9, max_depth=20, n_estimators=100, verbose=4) # tree
	clf = LogisticRegression()
	real(args, clf).to_csv(REAL_PATH+'submission_ensemble_{}.csv'.format(''.join(args)), header=True)

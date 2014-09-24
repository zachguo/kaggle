#!/usr/bin/env python
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = list(set(list(train.columns.values)) - set(['is_exciting', 'projectid']))
xtest = test[features]

var_cat = ['primary_focus_subject', 'teacher_prefix', 'grade_level']
var_cat_group1 = ['state_region', 'school_metro']
var_cat_group2 = ['primary_focus_area', 'resource_type']
var_num = list(set(features) - set(var_cat+var_cat_group1+var_cat_group2))

def get_formula(l, sep=' + ', cat=False): 
	if cat: return sep.join(['C('+x+')' for x in l])
	else: return sep.join(l)

fml = 'is_exciting ~ ' + \
		get_formula(var_num) + \
		' + ' + \
		get_formula(var_cat) + \
		' + ' + \
		get_formula(var_cat_group1, ' * ') + \
		' + ' + \
		get_formula(var_cat_group2, ' * ')
print fml

clf = sm.GLM.from_formula(fml,data=train,family=sm.families.Binomial()).fit()
print clf.summary()

ypred = clf.predict()
print 'AUC:', roc_auc_score(list(train['is_exciting']), ypred)
test['is_exciting'] = clf.predict(xtest)
test[['projectid', 'is_exciting']].to_csv('submission.csv', index=False)
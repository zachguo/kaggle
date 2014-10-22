from run import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import NuSVC
from collections import defaultdict

def cv_clf(clf, df_train, df_test):
	clf.fit(df_train[cols], df_train['Label'].values)
	return clf.score(df_test[cols], df_test['Label'].values)

def kfold(df):
	for train_idx, test_idx in StratifiedKFold(df['Label'], 3):
		df_train = df.iloc[train_idx]
		df_test = df.iloc[test_idx]
		yield df_train, df_test

def cv_rf(df):
	cv_scores = defaultdict(list)
	for df_train, df_test in kfold(df):
		for n_est in [50, 75, 100]:
			for criterion in ['entropy']:
				params = (n_est, criterion)
				score = cv_clf(RandomForestClassifier(n_estimators=n_est, criterion=criterion), df_train, df_test)
				cv_scores[params].append(score)
	return cv_scores

def cv_gbc(df):
	cv_scores = defaultdict(list)
	for df_train, df_test in kfold(df):
		for n_est in [100, 200]:
			for m_d in [7]:
				for m_s_l in [100]:
					for m_f in ['log2', 'auto', None]:
						params = (n_est, m_d, m_s_l, m_f)
						score = cv_clf(GradientBoostingClassifier(n_estimators=n_est, max_depth=m_d, min_samples_leaf=m_s_l, max_features=m_f), df_train, df_test)
						cv_scores[params].append(score)
	return cv_scores

def cv_svc(df):
	cv_scores = defaultdict(list)
	for df_train, df_test in kfold(df):
		params = ()
		score = cv_clf(NuSVC(nu=0.5, kernel='poly', verbose=True), df_train, df_test)
		cv_scores[params].append(score)
	return cv_scores

def cv_submodels(train, cols):
	for name, df in train.groupby('999'):
		scaler = MinMaxScaler()
		df[cols] = scaler.fit_transform(df[cols])
		df.to_csv('data/train_%d.csv' % name)
		print 'Submodel', nnnpatterns[int(name)]
		cv_scores = cv_gbc(df)
		# cv_scores = cv_rf(df)
		# cv_scores = cv_svc(df)
		for params, scores in sorted(cv_scores.items()):
			print params, ':', scores

cv_submodels(train, cols)

# for GBC, it seems that 100,7,100,log2 or 200,7,100,auto works best
# for RF, 100,entropy
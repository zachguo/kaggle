#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Siyuan Guo, June 2014"""

import pandas as pd
import coldocdist as cdd
from sklearn import preprocessing as pp
import math, random, states



class Data(object):

	NUM_COLS = [u'fulfillment_labor_materials', 
				u'total_price_excluding_optional_support', 
				u'total_price_including_optional_support', 
				u'students_reached']
	CAT_COLS = [u'school_state',
				u'school_metro', 
				u'teacher_prefix', 
				u'primary_focus_subject', 
				u'primary_focus_area', 
				u'secondary_focus_subject', 
				u'secondary_focus_area', 
				u'resource_type', 
				u'poverty_level', 
				u'grade_level']
	BOOL_COLS = [u'school_charter', 
				u'school_magnet', 
				u'school_year_round', 
				u'school_nlns', 
				u'school_kipp', 
				u'school_charter_ready_promise',
				u'teacher_teach_for_america', 
				u'teacher_ny_teaching_fellow', 
				u'eligible_double_your_impact_match', 
				u'eligible_almost_home_match']
	TEXT_COLS = [u'title', 
				u'need_statement', 
				u'short_description', 
				u'essay']
	USELESS_COLUMNS = [u'school_latitude', 
				u'school_city',  
				u'school_zip',
				u'school_longitude', 
				u'school_district', 
				u'school_county', 
				u'teacher_acctid_x', 
				u'teacher_acctid_y', 
				u'schoolid', 
				u'school_ncesid']

	def __init__(self):
		self.ctmatrix = None
		self.train, self.test = self.prepare()

	def prepare(self):
		# projects.csv / essays.csv has 664098 unique projects.
		data = pd.merge(pd.read_csv('projects.csv'), pd.read_csv('essays.csv'), on='projectid')

		print '  Preprocessing...'
		# remove data for projects posted before 2010
		data = data[data['date_posted'] >= '2010-06-01']
		# deal with missing values
		data['students_reached'] = data['students_reached'].fillna(30.0) #other numeric vars dont have NaN
		data[u'teacher_prefix'] = data[u'teacher_prefix'].fillna('Mrs.')
		data[self.CAT_COLS] = data[self.CAT_COLS].fillna('Unknown')
		data.dropna(subset=self.TEXT_COLS, inplace=True) # no nan in text columns in test data
		# drop useless columns
		data.drop('secondary_focus_subject', inplace=True, axis=1)
		data.drop('secondary_focus_area', inplace=True, axis=1)
		self.CAT_COLS.remove('secondary_focus_subject')
		self.CAT_COLS.remove('secondary_focus_area')
		# deal with outliers, remove all rows of which students_reached>1000, tpeos>45000, or tpios>50000.
		data = data[data['students_reached'] <= 1000]
		data = data[data['total_price_excluding_optional_support'] <= 45000]
		data = data[data['total_price_including_optional_support'] <= 50000]
		# train contains zero students_reached, but not test
		data = data[data['students_reached'] != 0]
		# reduce imbalanced sample
		data_out = pd.merge(data, pd.read_csv('outcomes.csv')[['projectid', 'is_exciting']], on='projectid', how='left')['is_exciting']
		index_fat = data_out[data_out=='f'].index
		index_remove = random.sample(index_fat, len(index_fat)-(data_out=='t').sum())
		assert len(index_fat)-len(index_remove) == len(data_out)-len(data_out[data_out.isnull()])-len(index_fat)
		data.drop(set(data.index) & set(index_remove), inplace=True)
		del data_out, index_fat, index_remove
		# normalize numeric vars, convert power distribution to normal distribution
		# data[self.NUM_COLS] = data[self.NUM_COLS].applymap(lambda x:math.log(x+0.01))
		# set proper dtype for columns
		for col in self.TEXT_COLS:
			data[col] = data[col].astype(str)

		print '  Deriving features...'
		data = self.derive_features(data)

		print '  Split training and testing sets...'
		# split train and test set
		train_idx = data['date_posted'] < '2014-01-01'
		test_idx = data['date_posted'] >= '2014-01-01'
		data.drop(u'date_posted', inplace=True, axis=1)
		train = data[train_idx]
		test = data[test_idx]
		del data #release memory
		train = pd.merge(train, pd.read_csv('outcomes.csv')[['projectid', 'is_exciting', 'fully_funded']].applymap(lambda x:{'t':1, 'f':0}[x] if x=='t' or x=='f' else x), on='projectid')
		# , 'great_chat'
		# outcomes.csv has 619326 unique projects, all posted before 2014-1-1.

		def get_history(col_teacherid_train, col_target, col_teacherid_test):
			count_dict = {}
			history = []
			for teacher_id,outcome in zip(col_teacherid_train, col_target):
				if teacher_id in count_dict:
					history.append(count_dict[teacher_id])
					count_dict[teacher_id] += outcome
				else:
					history.append(0)
					count_dict[teacher_id] = 1
			return history, col_teacherid_test.map(lambda x:count_dict[x] if x in count_dict else 0)

		def get_rate(col_numer, col_denom):
			return col_numer.div(col_denom + 0.001)

		train['num_submitted_projects'], test['num_submitted_projects'] = get_history(train['teacher_acctid_x'], [1]*len(train['teacher_acctid_x']), test['teacher_acctid_x'])
		train['num_is_exciting'], test['num_is_exciting'] = get_history(train['teacher_acctid_x'], train['is_exciting'], test['teacher_acctid_x'])
		train['num_funded_projects'], test['num_funded_projects'] = get_history(train['teacher_acctid_x'], train['fully_funded'], test['teacher_acctid_x'])
		# train['num_great_chat'], test['num_great_chat'] = get_history(train['teacher_acctid_x'], train['great_chat'], test['teacher_acctid_x'])

		train['exciting_rate'] = get_rate(train['num_is_exciting'], train['num_submitted_projects'])
		test['exciting_rate'] = get_rate(test['num_is_exciting'], test['num_submitted_projects'])
		train['funded_rate'] = get_rate(train['num_funded_projects'], train['num_submitted_projects'])
		test['funded_rate'] = get_rate(test['num_funded_projects'], test['num_submitted_projects'])
		train['funded2exciting_rate'] = get_rate(train['num_is_exciting'], train['num_funded_projects'])
		test['funded2exciting_rate'] = get_rate(test['num_is_exciting'], test['num_funded_projects'])

		# train['greatchat_rate'] = get_rate(train['num_great_chat'], train['num_submitted_projects'])
		# test['greatchat_rate'] = get_rate(test['num_great_chat'], test['num_submitted_projects'])
		# train['greatchat2exciting_rate'] = get_rate(train['num_is_exciting'], train['num_great_chat'])
		# test['greatchat2exciting_rate'] = get_rate(test['num_is_exciting'], test['num_great_chat'])

		# including following features will detriment performance by 0.001
		train['num_submitted_projects_school'], test['num_submitted_projects_school'] = get_history(train['schoolid'], [1]*len(train['schoolid']), test['schoolid'])
		train['num_is_exciting_school'], test['num_is_exciting_school'] = get_history(train['schoolid'], train['is_exciting'], test['schoolid'])
		train['num_funded_projects_school'], test['num_funded_projects_school'] = get_history(train['schoolid'], train['fully_funded'], test['schoolid'])
		train['exciting_rate_school'] = get_rate(train['num_is_exciting_school'], train['num_submitted_projects_school'])
		test['exciting_rate_school'] = get_rate(test['num_is_exciting_school'], test['num_submitted_projects_school'])
		train['funded_rate_school'] = get_rate(train['num_funded_projects_school'], train['num_submitted_projects_school'])
		test['funded_rate_school'] = get_rate(test['num_funded_projects_school'], test['num_submitted_projects_school'])
		train['funded2exciting_rate_school'] = get_rate(train['num_is_exciting_school'], train['num_funded_projects_school'])
		test['funded2exciting_rate_school'] = get_rate(test['num_is_exciting_school'], test['num_funded_projects_school'])

		# drop useless columns
		for col in self.USELESS_COLUMNS:
			train.drop(col, inplace=True, axis=1)
			test.drop(col, inplace=True, axis=1)
		for col in [u'fully_funded']:#, 'great_chat'
			train.drop(col, inplace=True, axis=1)

		print '  ', train.shape, test.shape

		return train, test

	def output(self):
		"""Output data into CSVs"""
		self.train.to_csv('train.csv', index=False)
		self.test.to_csv('test.csv', index=False)

	def derive_features(self, data):
		# engineer new numeric vars
		data[u'expense_optional_support'] = data[u'total_price_including_optional_support'] - data[u'total_price_excluding_optional_support']
		data[u'expense_optional_support_percentage'] = data[u'expense_optional_support'].div(data[u'total_price_including_optional_support'])
		data[u'expense_per_student'] = data[u'total_price_including_optional_support'].div(data[u'students_reached'])
		self.NUM_COLS += [u'expense_optional_support', u'expense_optional_support_percentage', u'expense_per_student']
		# numerify poverty_level
		data[u'poverty_level'] = data[u'poverty_level'].map({'low poverty':1., 'moderate poverty':2., 'high poverty':3., 'highest poverty':4.})
		self.CAT_COLS.remove(u'poverty_level')
		self.NUM_COLS.append(u'poverty_level')
		# transform school_state
		data[u'state_no_internet_rate'] = data[u'school_state'].map(states.NONETRATE)
		data[u'state_region'] = data[u'school_state'].map(states.REGION)
		data.drop(u'school_state', inplace=True, axis=1)
		self.CAT_COLS.remove(u'school_state')
		self.CAT_COLS.append(u'state_region')
		self.NUM_COLS.append(u'state_no_internet_rate')
		# # engineer new categotical vars
		# data[u'subject_x_resource'] = data['primary_focus_subject'] + '_' + data['resource_type']
		# data[u'metro_x_poverty'] = data[u'school_metro'] + ' ' + data[u'poverty_level']
		# self.CAT_COLS += ['subject_x_resource', u'metro_x_poverty']
		# for col in [u'primary_focus_subject', u'resource_type', u'school_metro', u'poverty_level']:
		# 	self.CAT_COLS.remove(col)
		# 	data.drop(col, inplace=True, axis=1)
		# engineer new boolean vars
		data[u'whether_ncesid'] = data[u'school_ncesid'].isnull()
		self.BOOL_COLS.append(u'whether_ncesid')
		# seasonal pattern?
		data[u'whether_mildseason'] = data[u'date_posted'].apply(lambda x:'-05-' in x or '-06-' in x or '-04-' in x)
		self.BOOL_COLS.append(u'whether_mildseason')
		# numerify boolean vars
		for col in self.BOOL_COLS:
			data[col] = data[col].map({'t':1, 'f':0, True:1, False:0})
		# # dummify categorical vars
		# for col in self.CAT_COLS:
		# 	data = pd.concat([data, pd.get_dummies(data[col], prefix=''.join([x[0] for x in col.split('_')]))], axis=1)
		# 	data.drop(col, inplace=True, axis=1)
		# derive text features
		for col in self.TEXT_COLS:
			data = self.get_simple_text_features(col, data)
		data = self.get_topic_features('need_statement', data)
		data['desc_essay'] = data['short_description'] + data['essay']
		data = self.get_topic_features('desc_essay', data)
		for col in self.TEXT_COLS+['desc_essay']:
			data.drop(col, inplace=True, axis=1)
		# include newly created numeric vars
		f = lambda s:s.endswith('_length') or s.endswith('_num_qmark') or s.endswith('_num_emark') or s.startswith('topic_')
		self.NUM_COLS += filter(f, data.columns)
		# feature scaling
		# data[self.NUM_COLS] = pp.normalize(data[self.NUM_COLS])
		data[self.NUM_COLS] = pp.StandardScaler().fit_transform(data[self.NUM_COLS])
		return data

	def get_simple_text_features(self, col, data):
		print '	Deriving text features for %s' % col
		data[col+'_length'] = data[col].apply(len)
		data[col+'_num_qmark'] = data[col].apply(lambda s: s.count("?")).div(data[col+'_length'])
		data[col+'_num_emark'] = data[col].apply(lambda s: s.count("!")).div(data[col+'_length'])
		return data

	def get_topic_features(self, col, data):
		print '	Deriving topic features for %s' % col
		from string import punctuation
		from gensim import corpora, models
		from nltk.corpus import stopwords
		from nltk.stem.porter import PorterStemmer
		stemmer = PorterStemmer()

		print data[col]
		print set(data[col].apply(type))

		def get_words(text):
			return [stemmer.stem(w.strip(punctuation)) for w in text.lower().split() if w.strip(punctuation) not in stopwords.words('english')]

		# generate topics for corpora
		texts = data[col].apply(get_words)
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=20, passes=2, iterations=50)
		lda.print_topics(-1)

		# get topic distribution for doc
		def get_topics(words):
			return dict(lda[dictionary.doc2bow(words)])

		topics_df = pd.DataFrame(texts.apply(get_topics).tolist()).fillna(0.001)
		topics_df.columns = ['topic_'+str(cn)+'_'+col for cn in topics_df.columns]
		return pd.merge(data, topics_df, left_index=True, right_index=True)

if __name__ == '__main__':
	print "Creating data..."
	Data().output()
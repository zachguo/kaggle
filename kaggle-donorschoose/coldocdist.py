#!/usr/bin/env python

import pandas as pd
import nltk.data as nd
from math import log, log10, sqrt, pow
from collections import defaultdict
from string import punctuation, whitespace
from functools import partial

WASTE = punctuation+whitespace
SENT_DETECTOR = nd.load('tokenizers/punkt/english.pickle')

# Helpers

def freq2prob(tfdict, pool):
	"""convert a dictionary of raw frequencies to probabilities"""
	tfdict = {term:tfdict[term] for term in tfdict if term in pool}
	total = float(sum(tfdict.values()))
	return {t:tfdict[t]/total for t in tfdict}

# Core

def get_ngrams(s, n, t):
	"""
	@param s, a piece of text
	@param n, length of ngrams to be extracted
	@param t, threshold frequency for a term to be included.
	@return ngram_dict, a dict of ngram-freq pair
	"""
	ngram_dict = defaultdict(int)
	# segment sentences
	sents = SENT_DETECTOR.tokenize(s)
	for sent in sents:
		# split sub-sentences, simply use ', ' as separator
		for subsent in sent.split(', '):
			subsent = [w.strip(WASTE).lower() for w in subsent.split()]
			for i in range(len(subsent)-n+1):
				ngram_dict[' '.join(subsent[i:i+n])] += 1
	return dict([(term,freq) for term,freq in ngram_dict.iteritems() if freq >= t])

def compute_cce(ctmatrix):
	"""
	Compute cross-collection entropy for each term.

	@param ctmatrix, a pandas dataframe representing term * label matrix.
	@return a dictionary of cross-collection entropies (term as key):
	        {u'murwara': 0.9999989777855017, 
	         u'fawn': 0.8813360127166802,
	         ... }
	"""

	# print "      Generating cross-collection entropy..."
	# Normalize each row from freq to prob
	ctmatrix = ctmatrix.div(ctmatrix.sum(axis=1), axis=0)
	# compute cross-collection entropy and return it, 12 is number of chronons.
	ctmatrix = ctmatrix.applymap(lambda x: x*log(x)).sum(axis=1)
	return ctmatrix.apply(lambda e: 1+1/log(12)*e).to_dict()

def compute_llr(ctmatrix):
	"""
	Compute log( p(w|col) / p(w/C) ), where col is collection and C is corpora.

	@param ctmatrix, a pandas dataframe representing term * collection matrix.
	@return a 2D dictionary in format of {'pre-1839':{'term': 0.003, ...}, ...}
	"""

	# Normalize each column from freq to prob: p(w|col)
	tfcollection = ctmatrix.div(ctmatrix.sum(axis=0), axis=1)
	# Sum all columns into one column then convert from freq to prob: p(w|C)
	tfcorpora = ctmatrix.sum(axis=1)
	tfcorpora = tfcorpora.div(tfcorpora.sum(axis=0))
	# Compute log likelihood ratio
	llrmatrix = tfcollection.div(tfcorpora, axis=0).applymap(log)
	return llrmatrix.to_dict()

def compute_nllr(dtmatrix, ccedict, llrdict):
	"""Compute NLLR score"""
	nllrdict = {}
	for i in dtmatrix.index:
		probs = freq2prob(dtmatrix[i], ccedict.keys())
		nllrdict[i] = {}
		for label in llrdict.keys():
			nllrdict[i][label] = sum([ccedict[term] * probs[term] * llrdict[label][term] for term in probs])
	return pd.DataFrame.from_dict(nllrdict, orient='index')

def get_dtmatrix(col_text, n, t):
	"""
	Convert a dataframe column of text into a column of ngram term-frequencies
	@param col_text, a pandas dataframe column of text.
	@param n, n of ngram
	@return dtmatrix, document-term matrix
	"""
	#fill NaN with empty string
	col_text.fillna("", inplace=True)
	#transform each text cell into term freq dict
	dtmatrix = col_text.apply(lambda x:get_ngrams(x, n, t))
	return dtmatrix

def get_ctmatrix_n_dtmatrix(col_text, col_label, n, t):
	"""
	@param col_text, a pandas dataframe column of text.
	@param col_label, a pandas dataframe column of labels.
	@param n, n of ngram
	@return ctmatrix, collection-term matrix
	@return dtmatrix, document-term matrix
	"""
	assert all(col_text.index == col_label.index)
	dtmatrix = get_dtmatrix(col_text, n, t)
	#aggregate all term freq dict based on labels
	ctdict = {l:defaultdict(int) for l in set(col_label)}
	for i in dtmatrix.index:
		l = col_label[i]
		tf = dtmatrix[i]
		for k in tf:
			ctdict[l][k] += tf[k]
	ctmatrix = pd.DataFrame(ctdict).fillna(0.0001)
	return ctmatrix, dtmatrix

def get_cdmatrix(ctmatrix, dtmatrix, suffix):
	"""
	@param ctmatrix, collection-term matrix
	@param dtmatrix, document-term matrix
	@param suffix, a suffix string to be added to column names
	@return cdmatrix, a pandas dataframe with label as column names and 
	                  term as row index, each cell is CCEwNLLR score.
	"""
	# compute cross-collection entropy
	ccedict = compute_cce(ctmatrix)
	# compute log-likelihood ratio
	llrdict = compute_llr(ctmatrix)
	# compute cross-collection-entropy weighted normalized-log-likelihood ratio
	cdmatrix = compute_nllr(dtmatrix, ccedict, llrdict)
	cdmatrix.columns = [str(cname)+'_'+str(suffix) for cname in cdmatrix.columns]
	return cdmatrix

def test():
	print get_ngrams('Hello world, this is a test text, this is another one.', 1, 1)
	print get_ngrams('Hello world, this is a test text, this is another one.', 2, 1)
	print get_ngrams('Hello world, this is a test text, this is another one.', 3, 1)
	# x = pd.read_csv('testtext.csv')
	# print x
	# ctmatrix, dtmatrix = get_ctmatrix_n_dtmatrix(x['essay'], x['is_exciting'], 1)
	# print pd.concat([x, get_cdmatrix(ctmatrix, dtmatrix, 'uni')], axis=1)

if __name__ == '__main__':
	test()
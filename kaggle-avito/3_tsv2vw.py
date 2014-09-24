#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert tsv to VW input format"""

import csv, sys, json, re
from numpy import log
from string import punctuation
from nltk.corpus import stopwords
from nltk import SnowballStemmer

# Constants & Helpers

engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))
stopwords= frozenset(word.decode('utf-8') for word in stopwords.words("russian") if word!="не")
stemmer = SnowballStemmer('russian')
NONFEATURES = ['itemid', 'category', 'subcategory', 'is_blocked', 'is_proved', 'close_hours', 'title', 'description', 'attrs']
NUM_FEATURES = ['price', 'phones_cnt', 'emails_cnt', 'urls_cnt']

def correct_word(w):
	""" Corrects word by replacing characters with written similarly depending on which language the word. 
		Fraudsters use this technique to avoid detection by anti-fraud algorithms."""
	num_eng = len(re.findall(u"[а-я]",w))
	num_rus = len(re.findall(u"[a-z]",w))
	if num_eng and num_rus:
		if num_eng > num_rus: return w.translate(eng_rusTranslateTable)
		else: return w.translate(rus_engTranslateTable)
	else:
		return w

def get_words(text): 
	text = re.sub(u"[^0-9a-zа-я]", ' ', text.decode('utf-8').lower())
	return [w.strip(punctuation) for w in text.split()]

def get_bigrams(wordlist):
	return ' '.join(['__'.join(wordlist[i:i+2]).encode('utf-8') for i in range(len(wordlist)-1)])

def get_trigrams(wordlist):
	return ' '.join(['__'.join(wordlist[i:i+3]).encode('utf-8') for i in range(len(wordlist)-2)])

def count_fraudwords(wordlist):
	return float(sum([1 if re.search(u"[a-z]", w) and re.search(u"[а-я]", w) else 0 for w in wordlist])) if wordlist else 0.

def filter_n_correct_words(wordlist):
	return [correct_word(w) for w in wordlist if len(w) > 1 and w not in stopwords]

def stem_words(wordlist):
	return [w if re.search(u"[0-9a-z]", w) else stemmer.stem(w) for w in wordlist] # only stem russian words

def featurify(line):
	features = []

	try:
		label = line['is_blocked']
	except KeyError:
		label = '1'
	if label == '0': label = '-1' #logistic vw only accept 1/-1 labels

	features.append("{} 1 {}|i".format(label, line['itemid']))
	features.append("{}:{}".format('price', log(float(line['price']) + 0.01)))

	# text features
	for c in ('title', 'description'):
		wordlist = get_words(line[c])
		length = len(wordlist)
		bigrams = get_bigrams(wordlist)
		trigrams = get_trigrams(wordlist)
		num_fraud = count_fraudwords(wordlist)/(length+0.01)
		wordlist = filter_n_correct_words(wordlist)
		# wordlist = stem_words(wordlist)
		v = ' '.join([w.encode('utf-8') for w in wordlist] + [bigrams] + [trigrams])
		features.append("|{} num_fraud:{} {}".format(c[0], num_fraud, v))

	# more numeric features
	for c in ('phones_cnt', 'emails_cnt', 'urls_cnt'):
		v = float(line[c])/(length+0.01)
		features.append("{}:{}".format(c, v))
		
	# categorical features	
	for c in ('category', 'subcategory'):
		v = line[c].replace(' ', '_')
		features.append("|{} {}".format(c[0], v))
	
	# unwind attrs as text
	attrs = None
	if line['attrs']:
		attrs = eval(line['attrs'].replace('/"', '\'').replace("'}", '"}'))
	if attrs:
		attribs = []
		for k, v in attrs.items():
			attribs.append((k + '_' + v).replace(' ', '_').replace(':', '_'))
		attribs = [x.encode('utf-8') for x in attribs]
		features.append("|a {}".format(" ".join(attribs)))
		
	features = " ".join(features)
	features += "\n"
	return features


if __name__ == '__main__':
	input_file, output_file = sys.argv[1:3]
	reader = csv.DictReader(open(input_file, 'rb'), delimiter = '\t')
	with open(output_file, 'wb') as fout:
		for line in reader:	
			newline = featurify(line)
			fout.write(newline)
	
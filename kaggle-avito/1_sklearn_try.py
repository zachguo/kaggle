#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Avito: Sklearn KNN (not used)"""

import pandas as pd
import re, sys
from numpy import isfinite, asarray, log
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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

def count_fraudwords(wordlist):
    return float(sum([1 if re.search(u"[a-z]", w) and re.search(u"[а-я]", w) else 0 for w in wordlist])) if wordlist else 0.

def filter_n_correct_words(wordlist):
    return [correct_word(w) for w in wordlist if len(w) > 1 and w not in stopwords]

def stem_words(wordlist):
    return [w if re.search(u"[0-9a-z]", w) else stemmer.stem(w) for w in wordlist] # only stem russian words

def feature_scaling(df):
    return StandardScaler().fit_transform(df)

def unwind(col, fillna='Unknown', dummy=True):
    """
    Unwind a column of dict strings into a dataframe of categorical variables
    @param col, a pandas series of dict strings
    @return df, a pandas dataframe of categorical variables
    """
    df = pd.DataFrame(col.fillna('{}').astype(str).apply(lambda x:eval(x.replace('/"', '\'').replace("'}", '"}'))).tolist())
    # remove cols with too many NaNs (95% are NaN), or better way to compress categorical vars?
    cols_wtmn = [k for k,v in df.isnull().sum(axis=0).apply(lambda x:x>=df.shape[0]*0.95).to_dict().items() if v]
    if cols_wtmn: 
        print '  Deleted '+', '.join(cols_wtmn)
        for colname_wtmn in cols_wtmn:
            df.drop(colname_wtmn, axis=1, inplace=True)
    # fillna
    if fillna is not None: df = df.fillna(fillna)
    # handle variables with too many classes
    # delete columns with most classes are unique, like 'address', or treating it as a text var?
    if isinstance(df, pd.Series):
        if df.value_counts().shape[0] > df.shape[0]/2: df = pd.Series()
    elif isinstance(df, pd.DataFrame):
        for colname_df in df.columns:
            if df[colname_df].value_counts().shape[0] > df.shape[0]/2:
                print '  Deleted ' + colname_df
                df.drop(colname_df, axis=1, inplace=True)
    # only keep first n_keep most frequent classes, or better way to compress classes?
    n_keep = 30
    if isinstance(df, pd.Series): df = df.replace(list(df.value_counts()[n_keep:].index), 'Rare') if list(df.value_counts()[n_keep:].index) else df
    elif isinstance(df, pd.DataFrame): 
        df = df.apply(lambda x:x.replace(list(x.value_counts()[n_keep:].index), 'Rare') if list(x.value_counts()[n_keep:].index) else x, axis=0)
    # dummify
    if dummy and not df.empty: 
        if isinstance(df, pd.Series): df = pd.get_dummies(df, prefix=df.name+'_')
        elif isinstance(df, pd.DataFrame): 
            for colname_df in df.columns:
                dummies = pd.get_dummies(df[colname_df], prefix=colname_df+'_')
                df.drop(colname_df, axis=1, inplace=True)
                for colname_dm in dummies.columns:
                    df[colname_dm] = dummies[colname_dm].values
        else: print 'Invalid df to unwind.'
    return df

def get_simple_text_features(col):
    colname = col.name
    col = col.astype(str).apply(get_words)
    length = col.apply(len).apply(float)
    fraudwords_cnt = col.apply(count_fraudwords)
    fraudwords_exist = (fraudwords_cnt > 0).map({True:1, False:0})
    # correct, stem and concate cleaned text
    col = col.apply(filter_n_correct_words).apply(stem_words).apply(lambda x:' '.join([w.encode('utf-8') for w in x]))
    df = pd.DataFrame({colname: col.values})
    # add fraudwords features
    df['length_'+colname] = feature_scaling(length.values)
    df['fraudwords_cnt_'+colname] = feature_scaling(fraudwords_cnt.values)
    df['fraudwords_exist_'+colname] = fraudwords_exist.values
    return df

def get_tfidf_features(col):
    from sklearn.feature_extraction.text import TfidfVectorizer
    colname = col.name
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), strip_accents='unicode', max_features=500, norm='l2')
    X_tfidf = vectorizer.fit_transform(col).todense()
    df = pd.DataFrame(X_tfidf, index=col.index)
    df.columns = ['tfidf_'+str(cn)+'_'+colname for cn in df.columns]
    return df

def featurify(df, f, colname, drop=True):
    features = f(df[colname])
    if drop: df.drop(colname, axis=1, inplace=True)
    for column in features.columns:
        df[column] = features[column].values
    if len(features.columns)>0: 
        assert all(all(df[fc].values == features[fc].values) for fc in features.columns)
    return df

def derive_attrs(df):
    print '    Deriving \'attrs\' features ...'
    return featurify(df, unwind, 'attrs')

def derive_text(df):
    print '    Deriving simple text features, cleaning text ...'
    df = featurify(df, get_simple_text_features, 'title')
    df = featurify(df, get_simple_text_features, 'description')
    return df

def derive_tfidf(df):
    print '    Deriving tfidf features ...' 
    df = featurify(df, get_tfidf_features, 'title')
    df = featurify(df, get_tfidf_features, 'description')
    return df

def cv_model(df, clf):
    X = df[list(set(df.columns) - set(NONFEATURES))].to_sparse()
    y = asarray(df['is_blocked'].tolist())
    print '    Features size:', X.shape
    assert all(X.applymap(isfinite).apply(all, axis=0))
    cv_scores = cross_val_score(clf, X, y, scoring='roc_auc').tolist()
    avg_cv_scores = float(sum(cv_scores))/len(cv_scores)
    return avg_cv_scores

def fit_model(df, clf):
    X = df[list(set(df.columns) - set(NONFEATURES))].to_sparse()
    y = asarray(df['is_blocked'].tolist())
    print '    Features size:', X.shape
    clf.fit(X,y)
    ypred = [x[clf.classes_.tolist().index(1)] for x in clf.predict_proba(X)]
    return ypred, clf

def predict_model(df, clf_trained):
    if df.empty: return []
    X = df[list(set(df.columns) - set(NONFEATURES))].to_sparse()
    print '    Features size:', X.shape
    return [x[clf_trained.classes_.tolist().index(1)] for x in clf_trained.predict_proba(X)]

def get_outcome(df, ypred, name):
    return pd.DataFrame({'itemid':df['itemid'].tolist(), 'sub_'+'_'.join(name).replace(' ', '_'):ypred})

def get_ensembled_data_cv(data, fillna=0.):
    data['price'] = data['price'].add(0.01).apply(log)
    grouped = data.groupby(['category', 'subcategory'])
    for name, df in grouped:
        print '  Deriving features for %s ...' % ' - '.join(name)
        df[NUM_FEATURES] = feature_scaling(df[NUM_FEATURES])
        df = derive_attrs(df)
        avg_cv_scores = cv_model(df, KNeighborsClassifier(125))
        print '    CV score is %f.' % avg_cv_scores
        if avg_cv_scores < 0.95:
            df = derive_text(df)
            df = derive_tfidf(df)
            avg_cv_scores_new = cv_model(df, KNeighborsClassifier(125))
            print '    CV scores improved by', avg_cv_scores_new - avg_cv_scores
        ypred, clf = fit_model(df, KNeighborsClassifier(125)) # clf unused in cross-validation
        outcome = get_outcome(df, ypred, name)
        data = pd.merge(data, outcome, on='itemid', how='left')
    if fillna is not None: data = data.fillna(fillna)
    data[NUM_FEATURES] = feature_scaling(data[NUM_FEATURES])
    return data

def get_ensembled_data_real(data, fillna=0.):
    # data is a dataframe with multiindex, which is different from cv version
    data['price'] = data['price'].add(0.01).apply(log)
    grouped = data.groupby(['category', 'subcategory'])
    for name, df in grouped:
        print '  Deriving features for %s ...' % ' - '.join(name)
        df[NUM_FEATURES] = feature_scaling(df[NUM_FEATURES])
        df = derive_attrs(df)
        df = derive_text(df)
        # df = derive_tfidf(df)
        ypred_train, clf_trained = fit_model(df.loc['train'], KNeighborsClassifier(125))
        ypred_test = predict_model(df.loc['test'], clf_trained)
        outcome = get_outcome(df.loc['train'], ypred_train, name).append(get_outcome(df.loc['test'], ypred_test, name)).reset_index()
        data = pd.merge(data, outcome, on='itemid', how='left')
    if fillna is not None: data = data.fillna(fillna)
    data[NUM_FEATURES] = feature_scaling(data[NUM_FEATURES])
    return data


# Workflow

def workflow_train(data_method, clf_method):

    if data_method == 'load': ## load data
        print "Loading data ..."
        train = pd.read_csv('train.tsv', sep='\t')
    elif data_method == 'create': ## Create data
        print "Creating data ..."
        train = pd.read_csv('avito_train.tsv', sep='\t')
        train = get_ensembled_data_cv(train)
        from csv import QUOTE_NONNUMERIC
        train.to_csv('train.tsv', sep='\t', quoting=QUOTE_NONNUMERIC)
    else:
        raise ValueError('data_method not valid')

    print "Ensembling models ..."
    if clf_method == 'LR':
        clf = LogisticRegression()
    elif clf_method == 'GBM':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(verbose=1)
    elif clf_method == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier()
    else:
        raise ValueError('clf_method not valid')

    features = list(set(train.columns) - set(NONFEATURES))
    print clf.fit(train[features], train['is_blocked'].tolist()).score(train[features], train['is_blocked'].tolist())


def workflow_real(data_method):
    if data_method == 'load': ## load data
        print "Loading data ..."
        data = pd.read_table('data.tsv', sep='\t')

    elif data_method == 'create': ## Create data
        print "Creating data ..."
        train = pd.read_csv('avito_train.tsv', sep='\t')
        test = pd.read_csv('avito_test.tsv', sep='\t')
        data = train.append(test).reset_index()
        new_index = pd.MultiIndex.from_tuples(zip(['train']*train.shape[0]+['test']*test.shape[0], data.index))
        data.index = new_index
        print data
        del train, test
        data = get_ensembled_data_real(data, fillna=0.)
        from csv import QUOTE_NONNUMERIC
        data.to_csv('data.tsv', sep='\t', quoting=QUOTE_NONNUMERIC, index=False)
    else:
        raise ValueError('data_method not valid')

    # remove unnecessary index
    data = data[filter(lambda x:not x.startswith('index'), data.columns)]
    # prepare train and test data for ensemble
    outcome_train = data[:3995803]['is_blocked'].tolist()
    itemid_test = pd.DataFrame(data[3995803:]['itemid'].tolist(), columns=['Id'])
    data = derive_tfidf(data)
    features = list(set(data.columns) - set(NONFEATURES))
    data = data[features]
    train = data[:3995803]
    test = data[3995803:]
    del data

    print 'Start ensembling ...'
    clf = LogisticRegression()
    clf.fit(train, outcome_train)
    print clf.score(train, outcome_train)
    print clf.classes_
    itemid_test['pred'] = [x[1] for x in clf.predict_proba(test)]
    itemid_test.sort('pred', ascending=0, inplace=True)
    itemid_test.to_csv('itemid_test.tsv', sep='\t')
    print itemid_test
    # print 'Outputting ...'
    # with open('submission.txt', 'w') as fout:
    #     fout.write('Id\n')
    #     for itemid in itemid_test['Id'].tolist():
    #         fout.write(str(itemid)+'\n')



if __name__ == '__main__':
    if sys.argv[1] == 'cv':
        workflow_train(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'real':
        workflow_real(sys.argv[2])

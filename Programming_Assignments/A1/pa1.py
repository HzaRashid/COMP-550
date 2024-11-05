''' IMPORT '''
# # download ntlk tokenizer, lemmatizer, & pos tagger:
# import nltk
# nltk.download('punkt', download_dir="./venv/lib/nltk_data")
# nltk.download('punkt_tab', download_dir="./venv/lib/nltk_data")
# nltk.download('wordnet', download_dir="./venv/lib/nltk_data")
# nltk.download('averaged_perceptron_tagger_eng', 
# download_dir="./venv/lib/nltk_data")
# #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer, PorterStemmer 
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import re
import os

''' PREPROCESS DATA:
Column 0: 'text'    (string)
Column 1: 'is_fact' (boolean)
'''
# FACTS
fact_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'./data/facts.txt'), sep='\t', names=['text'])
fact_data['is_fact'] = 1
# print(fact_data.head()) # verify

# FAKES
fake_data = pd.read_csv(os.path.join(os.path.dirname(__file__),'./data/fakes.txt'), sep='\t', names=['text'])
fake_data['is_fact'] = 0
# print(fake_data.head()) # verify

# STORE DATA IN ONE TABLE (shuffle handled later)
data = pd.concat([fact_data, fake_data])
# # verify
# print(data.head()) # should be 1 along 'is_fact'
# print(data.tail()) # should be 0 along 'is_fact'


''' 
Preprocessing Decisions:
    1. Convert text to lowercase, 
    2. Remove all but alphanumeric 
       & whitespace characters
    3. Tokenize (unigram)
    4. Lemmatize
    5. Stem
    6. POS tagging
    7. Term frequency feature vector
'''

data['text'] = data['text'].apply(
    lambda s: ''.join(re.findall("[a-zA-Z0-9 ]", s.lower())) 
    )
# # verify results
# print(data.head())
# print(data.tail())

''' 
given treeback pos tag
return wordnet pos tag
'''
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # default pos used by WordNetLemmatizer
    

lemmatize = WordNetLemmatizer().lemmatize
stem = PorterStemmer().stem

def lamda_stem(s):
   return ' '.join([stem(w) for w in s.split()])


def lamda_lemmatize(s):
    return' '.join([lemmatize(w) for w in s.split()])


def lambda_pos_lemmatize(s):
    return ' '.join([lemmatize(w, get_wordnet_pos(pos)) for w,pos in pos_tag(s.split())])


preprocess_methods = (
    ('STEM', lamda_stem),
    ('LEMMATIZE (Plain)', lamda_lemmatize),
    ('LEMMATIZE (With POS tags)', lambda_pos_lemmatize),
    )

# 70/10/20 train/validate/test split
train_ratio = 0.7
validation_ratio = 0.1
test_ratio = 0.2

train, valid_and_test = train_test_split(
    data,
    test_size=(1-train_ratio),
    shuffle=True,
    random_state=42
)
# reset indices after shuffle
train = train.reset_index(drop=True)
valid_and_test = valid_and_test.reset_index(drop=True)

''' Test all linear classifiers
    under all preprocessing methods

    The Classifiers:
    - Support Vector Machine (linear kernel)
    - Naive Bayes (Multinomial)
    - Logistic Regression
    - Linear Regression
'''
for name, proc_fn in preprocess_methods:
    print(f'{"="*20} {name}... {"="*20}')

    vectorizer = TfidfVectorizer(ngram_range=(1,3)) # Term Frequency

    X_train, y_train = ( 
        vectorizer.fit_transform(train['text'].apply(proc_fn)),
        train['is_fact']
    ) # fit training corpus and extract feature vectors

    X_valid_and_test, y_valid_and_test = (
        vectorizer.transform(valid_and_test['text'].apply(proc_fn)),
        valid_and_test['is_fact']
    ) # extract feature vectors w.r.t. training fit

    # num features obtained by current proprocessing method
    print('num features: ', X_train.shape[1], '\n')

    # get validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_and_test, 
        y_valid_and_test, 
        test_size=(test_ratio/(test_ratio + validation_ratio)),
        shuffle=True,
        random_state=42
        )
    

    ''' 
    Linear Support Vector Machine:
    '''

    ''' Validation:
        optimize the inverse-regularization parameter
    '''
    print(f"Validating linear SVM (optimize inverse-regularization parameter):")
    svm_scores = {}
    for reg_val in (1.0, 0.5, 0.25):
        svm_clf = SVC(kernel='linear', C=reg_val)
        svm_clf.fit(X_train, y_train)
        score = svm_clf.score(X_valid, y_valid)
        svm_scores[score] = reg_val
        print(f'C:={reg_val} -> Mean Accuracy: {score * 100}%')

    ''' Test
    '''
    opt_reg_svm = svm_scores[max(svm_scores)]
    print(f"\nTesting SVM (linear kernel, C:={opt_reg_svm}):")
    svm_clf = SVC(kernel='linear', C=opt_reg_svm)
    svm_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {svm_clf.score(X_test, y_test) * 100}%')


    ''' 
    Multinomial Naive Bayes:
    '''

    ''' Validation:
        optimize smoothing parameter
    '''
    print(f"\nValidating Multinomial Naive Bayes (optimize smoothing parameter):")
    nb_scores = {}
    for alpha in (1.0, 0.5, 0.25):
        nb_clf = MultinomialNB(alpha=alpha) 
        nb_clf.fit(X_train, y_train)
        score = nb_clf.score(X_valid, y_valid)
        nb_scores[score] = alpha
        print(f'alpha:={alpha} -> Mean Accuracy: {score * 100}%')

    ''' Test
    '''
    opt_alpha = nb_scores[max(nb_scores)]
    print(f"\nTesting Naive Bayes (Multinomial, alpha:={opt_alpha}):")
    nb_clf = MultinomialNB(alpha=opt_alpha) 
    nb_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {nb_clf.score(X_test, y_test) * 100}%')
    

    ''' Logistic Regression:
    '''

    ''' Validation:
        optimize inverse-regularization parameter
    '''
    print(f"\nValidating Logistic Regression (optimize inverse-regularization parameter):")
    lgr_scores = {}
    for reg_val in (1.0, 0.5, 0.25):
        lgr_clf = LogisticRegression(C=reg_val)
        lgr_clf.fit(X_train, y_train)
        score = lgr_clf.score(X_valid, y_valid)
        lgr_scores[score] = reg_val
        print(f'C:={reg_val} -> Mean Accuracy: {score * 100}%')

    ''' Test
    '''
    opt_reg_lgr = lgr_scores[max(lgr_scores)]
    print(f"\nTesting Logistic Regression (C:={opt_reg_lgr}):")
    lgr_clf = LogisticRegression(C=opt_reg_lgr) 
    lgr_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {lgr_clf.score(X_test, y_test) * 100}%')


    ''' Linear Regression
    '''
    print(f"\nLinear Regression:")
    linr_clf = LinearRegression() 
    linr_clf.fit(X_train, y_train)
    print(f'-> R^2: {linr_clf.score(X_test, y_test) * 100}%')

    print('\n')
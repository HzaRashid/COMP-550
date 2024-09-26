''' IMPORT '''
# # download ntlk tokenizer, lemmatizer, & pos tagger:
# import nltk
# nltk.download('punkt', download_dir="./venv/lib/nltk_data")
# nltk.download('punkt_tab', download_dir="./venv/lib/nltk_data")
# nltk.download('wordnet', download_dir="./venv/lib/nltk_data")
# nltk.download('averaged_perceptron_tagger_eng', 
# download_dir="./venv/lib/nltk_data")
# #
from sklearn.metrics import accuracy_score, classification_report
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

    

''' PREPROCESS DATA:
Column 0: 'text'    (string)
Column 1: 'is_fact' (boolean)
'''

# FACTS
fact_data = pd.read_csv('./data/test_facts.txt', sep='\t', names=['text'])
fact_data['is_fact'] = 1
# print(fact_data.head()) # verify

# FAKES
fake_data = pd.read_csv('./data/test_fakes.txt', sep='\t', names=['text'])
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

''' Feature extraction step:
    - map each row of 'text' column
      to its relative term frequency feature vector,
      where the corpus is the set
      of words in the entire dataset.

    - the 'is_fact' column is the true label.

    - n-gram range of 1-3 words is chosen,
      as city names, and names of historical 
      events/sites typically vary between 1-3 words.

    - The chosen vectorizer also downscales
      tokens that occur in many documents,
      which can help prevent the models
      from giving too much weight to 
      city names, offering normalization
      when working with multiple cities, and 
      thus handling outliers more effectively.
'''

vectorizer = TfidfVectorizer(ngram_range=(1,3)) # Term Frequency

lemmatize = WordNetLemmatizer().lemmatize
stem = PorterStemmer().stem

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

train_ratio = 0.6
validation_ratio = 0.1
test_ratio = 0.3


''' Train-test split:
    - train 60% of data, validate on 10%, test on rest.
    - shuffle data before hand.
'''
for name, proc_fn in preprocess_methods:
    print(f'{"="*20} {name}... {"="*20}')

    X = vectorizer.fit_transform(
        data['text'].apply(proc_fn)
        )
    y = data['is_fact'] # labels

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=(1-train_ratio),
        shuffle=True,
        random_state=42
        )
    
    # split test further to get validation set
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, 
        test_size=(test_ratio/(test_ratio + validation_ratio)),
        shuffle=True,
        random_state=42
        )

    ''' 
    APPLY LINEAR CLASSIFIERS:
        - Support Vector Machine 
        - Naive Bayes
        - Logistic Regression
    '''

    ''' Support Vector Machine 
        - linear kernel
        - validate training to optimize the
          regularization parameter
    '''

    ''' Validation 
    '''
    # regularization parameter: inversely proportional to regularization
    print(f"Validating SVM (linear kernel, optimize regularization parameter):")
    for reg_val in (1.0, 0.5, 0.25):
        print(f"C={reg_val}:")
        svm_clf = SVC(kernel='linear', C=reg_val)
        svm_clf.fit(X_train, y_train)
        print(f'-> Mean Accuracy: {svm_clf.score(X_valid, y_valid) * 100}%')

    ''' Test
    '''
    print(f"\nTesting SVM (linear kernel, C=0.5):")
    svm_clf = SVC(kernel='linear', C=0.5)
    svm_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {svm_clf.score(X_test, y_test) * 100}%')


    ''' Naive Bayes
        - Multinomial implementation
        - default configuration uses laplace smoothing
    '''

    ''' Validation 
    '''
    print(f"\nValidating Naive Bayes (Multinomial, optimize smoothing parameter")
    # smoothing parameter, compare Laplace with Lidstone
    for alpha in (1.0, 0.5, 0.25):
        print(f"alpha={alpha}:")
        nb_clf = MultinomialNB(alpha=alpha) 
        nb_clf.fit(X_train, y_train)
        print(f'-> Mean Accuracy: {nb_clf.score(X_valid, y_valid) * 100}%')

    ''' Test
    '''
    print(f"\nTesting Naive Bayes (Multinomial, alpha=0.5):")
    nb_clf = MultinomialNB(alpha=0.5) 
    nb_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {nb_clf.score(X_test, y_test) * 100}%')
    

    ''' Logistic Regression
        - validate training to optimize the
          regularization parameter
    '''

    ''' Validation 
    '''
    # regularization parameter: inversely proportional to regularization
    print(f"\nValidating Logistic Regression (optimize regularization parameter):")
    for reg_val in (1.0, 0.5, 0.25):
        print(f"C={reg_val}:")
        logr_clf = LogisticRegression(C=reg_val)
        logr_clf.fit(X_train, y_train)
        print(f'-> Mean Accuracy: {logr_clf.score(X_valid, y_valid) * 100}%')

    ''' Test
    '''
    print(f"\nTesting Logistic Regression (C=1.0):")
    logr_clf = LogisticRegression() 
    logr_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {logr_clf.score(X_test, y_test) * 100}%')


    ''' Linear Regression
    '''
    print(f"\nLinear Regression:")
    linr_clf = LinearRegression() 
    linr_clf.fit(X_train, y_train)
    print(f'-> R^2: {linr_clf.score(X_test, y_test) * 100}%')
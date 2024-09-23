''' IMPORT '''
import nltk
# # download ntlk tokenizer:
# nltk.download('punkt', download_dir="./venv/lib/nltk_data")
# nltk.download('punkt_tab', download_dir="./venv/lib/nltk_data")
# from nltk.tokenize import word_tokenize
# #
# # download ntlk lemmatizer:
# nltk.download('wordnet', download_dir="./venv/lib/nltk_data")
# #
# # download ntlk pos tagger:
# nltk.download('averaged_perceptron_tagger_eng', download_dir="./venv/lib/nltk_data")
# #

from nltk.stem import WordNetLemmatizer, PorterStemmer 
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import numpy as np
import pandas as pd
import re
''' usage 
print(WordNetLemmatizer().lemmatize('fairness'))
print(PorterStemmer().stem('fairness'))
# use tokenizer instead if punctuation included
print(pos_tag("this is a sentence".split()))
'''


# print(PorterStemmer().stem('fairness'))

''' PREPROCESS DATA:
Column 0: 'text'    (string)
Column 1: 'is_fact' (boolean)
'''

# FACTS
fact_data = pd.read_csv('./data/facts.txt', sep='\t', names=['text'])
fact_data['is_fact'] = 1
# print(fact_data.head()) # verify

# FAKES
fake_data = pd.read_csv('./data/fakes.txt', sep='\t', names=['text'])
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
    5. Term frequency feature vector

To lowercase:
    data['text'] = data['text'].apply(lambda s: s.lower())

Remove all but alphanumeric/whitespace chars:
    data['text'] = data['text'].apply(
        lambda s: ''.join(re.findall("[a-zA-Z0-9 ]", s)) 
        ) # re.findall preserves wspace

Tokenize: 
    # from nltk.tokenize import word_tokenize
    data['text'] = data['text'].apply(
        lambda s: ' '.join(word_tokenize(s))
        ) # word_tokenize removes wspace

Lemmatize:
    # lemmatize = WordNetLemmatizer().lemmatize
    data['text'] = data['text'].apply(
        lambda s: ' '.join([lemmatize(w) for w in s.split()])
        ) # split removes wspace
'''

''' All together: 
    - tokenization done implicitly (by re.findall),
      necessary for lemmatization.
    - feature extraction will also
      require tokens - sci-kit learn handles it.
'''

lemmatize = WordNetLemmatizer().lemmatize
data['text'] = data['text'].apply(
    lambda s: ''.join([
        lemmatize(w) for w in re.findall("[a-zA-Z0-9 ]", s.lower())
        ])
    )
# # verify results
# print(data.head())
# print(data.tail())


''' Feature extraction:
    - preprocessing thus far
      smooths out t.f. distribution;
      all lowercase, no punctuation, etc.
    - sklearn's feature extraction methods
      can also perform some of the preprocessing
      that has already been done.
'''

vectorizer = TfidfVectorizer() # Term Frequency

''' feature extraction step:
    - map each row of 'text' column
      to its t.f. feature vector,
      where the corpus is the set
      of words in the entire dataset.
    - the 'is_fact' column is the true label.
'''
X = vectorizer.fit_transform(data['text']) # inputs
y = data['is_fact'] # labels

# # dataset's text corpus
# print(vectorizer.get_feature_names_out())
# # given by: (dataset row length, dataset text corpus size)
# print(X.shape)

''' Train-test split:
    - train 80% of data, test on rest.
    - shuffle data before hand.
'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42, 
    shuffle=True
    )

''' APPLY LINEAR CLASSIFIERS:
'''

''' Support Vector Machine
'''
for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
    print(f"SVM with {kernel} kernel:")
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train, y_train)
    print(f'\tMean Accuracy: {svm_clf.score(X_test, y_test) * 100}%')
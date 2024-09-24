''' IMPORT '''
import nltk
# # download ntlk tokenizer, lemmatizer, & pos tagger:
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
import pandas as pd
import re


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
    5. Stem
    6. POS tagging
    7. Term frequency feature vector

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
      necessary for lemmatization/stemming/pos_tagging.
    - feature extraction will also
      require tokens - sci-kit learn handles it.
'''


data['text'] = data['text'].apply(
    lambda s: ''.join(re.findall("[a-zA-Z0-9 ]", s.lower())) 
    )
# # verify results
# print(data.head())
# print(data.tail())


''' Feature extraction comments:
    - preprocessing thus far
      smooths out t.f. distribution;
      all lowercase, no punctuation, etc.
    - sklearn's feature extraction methods
      can also perform some of the preprocessing
      that has already been done.
'''

vectorizer = TfidfVectorizer(ngram_range=(1,3)) # Term Frequency

''' feature extraction step:
    - map each row of 'text' column
      to its relative term frequency feature vector,
      where the corpus is the set
      of words in the entire dataset.
    - the 'is_fact' column is the true label.
    - n-gram range of 1-3 words is chosen,
      as a city name typically varies between
      1-3 words.
    - The chosen vectorizer also downscales
      tokens that occur in many documents,
      which can help normalize the data
      when working with multiple cities, and 
      thus handle outliers more effectively.
'''

lemmatize = WordNetLemmatizer().lemmatize
stem = PorterStemmer().stem

(X_lemmatize, X_stem, X_pos) = (
    vectorizer.fit_transform(
        data['text'].apply(
            lambda s: ' '.join([lemmatize(w) for w in s.split()])
            )
    ),
    vectorizer.fit_transform(
        data['text'].apply(
            lambda s: ' '.join([stem(w) for w in s.split()])
            )
        ),
    vectorizer.fit_transform(
        data['text'].apply(
            lambda s: ' '.join(['_'.join(t) for t in pos_tag(s.split())])
            )
        )
)

''' Train-test split:
    - train 80% of data, test on rest.
    - shuffle data before hand.
'''
for X, name in (
    (X_lemmatize, 'LEMMATIZE'), 
    (X_stem, 'STEM'), 
    (X_pos, 'POS')
    ):
    print(f'{"="*20} Testing {name}... {"="*20}')
    y = data['is_fact'] # labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.6,
        random_state=42, 
        shuffle=True
        )

    ''' 
    APPLY LINEAR CLASSIFIERS:
    '''

    ''' Support Vector Machine 
        - linear kernel
    '''
    print(f"SVM (linear kernel):")
    svm_clf = SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    print(f'-> Mean Accuracy: {svm_clf.score(X_test, y_test) * 100}%')


    ''' Naive Bayes
        - Multinomial implementation
        - default configuration uses laplace smoothing
    '''
    print(f"\nNaive Bayes (Multinomial):")
    nb_clf = MultinomialNB() 
    nb_clf.fit(X, y)
    print(f'-> Mean Accuracy: {nb_clf.score(X_test, y_test) * 100}%')


    ''' Logistic Regression
    '''
    print(f"\nLogistic Regression:")
    logr_clf = LogisticRegression() 
    logr_clf.fit(X, y)
    print(f'-> Mean Accuracy: {logr_clf.score(X_test, y_test) * 100}%')


    ''' Linear Regression
    '''
    print(f"\nLinear Regression:")
    linr_clf = LinearRegression() 
    linr_clf.fit(X, y)
    print(f'-> R^2: {linr_clf.score(X_test, y_test) * 100}%')
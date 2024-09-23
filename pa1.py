''' IMPORT '''
import nltk
# # uncomment upon first exec:
# nltk.download('punkt', download_dir="./venv/lib/nltk_data")
# nltk.download('punkt_tab', download_dir="./venv/lib/nltk_data")
# #
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import re


''' PREPROCESS DATA
Column 0: 'text'    (string)
Column 1: 'is_fact' (boolean)
'''

# FACTS
fact_data = pd.read_csv('./data/facts.txt', sep='\t', names=['text'])
fact_data['is_fact'] = 1
print(fact_data.head()) # verify

# FAKES
fake_data = pd.read_csv('./data/fakes.txt', sep='\t', names=['text'])
fake_data['is_fact'] = 0
print(fake_data.head()) # verify

# STORE DATA IN ONE TABLE
data = pd.concat([fact_data, fake_data])
# verify
print(data.head()) # should be 1 along 'is_fact'
print(data.tail()) # should be 0 along 'is_fact'


''' 
Preprocessing Decisions:
    1. Convert text to lowercase, 
    2. Remove all but alphanumeric 
       & whitespace characters
    3. Tokenize (unigram)
    4. Term frequency feature vector
'''

''' 
To lowercase:
    data['text'] = data['text'].apply(lambda s: s.lower())

Remove all but alphanumeric/whitespace chars:
    data['text'] = data['text'].apply(lambda s: ''.join(re.findall("[a-zA-Z0-9 ]", s)))

Tokenize: 
    data['text'] = data['text'].apply(lambda s: word_tokenize(s))
'''

''' All together: '''
data['text'] = data['text'].apply(
    lambda s: word_tokenize(''.join(re.findall("[a-zA-Z0-9 ]", s.lower())))
    )


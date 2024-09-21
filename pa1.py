''' IMPORT '''
import nltk
import sklearn
import numpy as np
import pandas as pd


''' PREPROCESS DATA 
Column 0: 'text'    (string)
Column 1: 'is_fact' (boolean)
'''

# FACTS
fact_data = pd.read_csv('./data/facts.txt', sep='\t')
fact_data['is_fact'] = 1
print(fact_data.head()) # verify

# FAKES
fake_data = pd.read_csv('./data/fakes.txt', sep='\t')
fake_data['is_fact'] = 0
print(fake_data.head()) # verify

''' tokenize '''


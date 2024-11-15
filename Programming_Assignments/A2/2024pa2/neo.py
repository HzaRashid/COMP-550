'''IMPORT'''
import os
from nltk.corpus import wordnet as wn
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
from flair.data import Sentence
from flair.nn import Classifier
import pandas as pd
from ast import literal_eval
import random 
# stop_words = set(stopwords.words('english'))
# tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation


''' Flair NER model '''
NERtagger = Classifier.load('ner-fast')

''' load data '''
data = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'pa2data.tsv'),
    converters={'label':literal_eval},
    sep='\t'
    )

train_data = data[data['id'].str.startswith('d001')]

''' MODELS '''
def eval_NEO(data):
    # Named Entity Overlap
    correct_prediction_ct = 0
    cur_sent = [None, None]
    for _, item in data.iterrows():
        item_lemma = item.lemma
        s = item.id.split('.')[1]
        if s != cur_sent[0]:
            cur_sent[0] = s
            cur_sent[1] = get_NE_tags(item.context.replace('_', ' '))

        synsets = wn.synsets(item_lemma)
        # gather definition and examples
        tagged_synsets = [
            (synset, get_NE_tags(synset.definition())) \
            if not synset.examples() else \
            (synset, get_NE_tags(random.choice(synset.examples())))
            for synset in synsets
            ]
        
        max_overlap_synset = get_max_NEO_synset(cur_sent[1], tagged_synsets)
        sense = None
        
        for lemma in max_overlap_synset[0].lemmas():
            if lemma.name().lower() == item.lemma.lower(): 
                sense = lemma.key()

        if sense in item.label:
            correct_prediction_ct += 1

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


# pick first synset every time
def eval_baseline(data):
    # Named Entity Overlap
    correct_prediction_ct = 0
    for _, item in data.iterrows():
        item_lemma = item.lemma

        synset = wn.synsets(item_lemma)[0]
        
        sense = None
        
        for lemma in synset.lemmas():
            if lemma.name().lower() == item.lemma.lower(): 
                sense = lemma.key()

        if sense in item.label:
            correct_prediction_ct += 1

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


''' HELPERS '''
def get_NE_tags(sentence):
    # run NER over sentence
    sentence = Sentence(sentence)
    NERtagger.predict(sentence)
    return [label.value for label in sentence.get_labels()]

def overlap_quantity(context1, context2):
    if len(context1) < len(context2):
        context1, context2 = context2, context1
    return len(set(context1).intersection(context2))

def get_max_NEO_synset(context, synsets):
    return max([
        (synset[0], overlap_quantity(context, synset[1])) \
        for synset in synsets
    ], key=lambda x: x[1])


if __name__ == "__main__":
    eval_NEO(train_data)


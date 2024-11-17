'''******************************************************************'''
'''******************************************************************'''
'''
INSTRUCTIONS:
    -  Scroll to the bottom (__name__ == "__main__")
    -  run test_model(model_name) with the model_name thats already there
    
'''
'''******************************************************************'''
'''******************************************************************'''
'''IMPORT'''
from nltk.corpus import wordnet as wn
from flair.data import Sentence
from flair.nn import Classifier
from ast import literal_eval
import pandas as pd
import random 
import os


''' load data '''
data = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'pa2data.tsv'),
    converters={'label':literal_eval},
    sep='\t'
    )

train_data = data[data['id'].str.startswith('d001')]
test_data = data[~data['id'].str.startswith('d001')]

''' MODELS '''
def eval_NEO(data, seq_tagger, st_name):
    ''' Named Entity Overlap '''
    correct_prediction_ct = 0
    cur_sent = [None, None] # document-id, Named-Entities list
    for _, item in data.iterrows():
        item_lemma = item.lemma
        s = item.id.split('.')[1]
        if s != cur_sent[0]:
            cur_sent[0] = s
            cur_sent[1] = get_NE_tags(item.context.replace('_', ' '), seq_tagger)

        # gather definition and examples
        synsets = wn.synsets(item_lemma)
        tagged_synsets = [
            (synset, get_NE_tags(synset.definition(), seq_tagger)) \
            if not synset.examples() else \
            (synset, get_NE_tags(random.choice(synset.examples()), seq_tagger))
            for synset in synsets
            ]
        
        sense = None
        max_overlap_synset = get_max_NEO_synset(cur_sent[1], tagged_synsets)
        for lemma in max_overlap_synset[0].lemmas():
            if lemma.name().lower() == item.lemma.lower(): 
                sense = lemma.key()

        if sense in item.label:
            correct_prediction_ct += 1

    print(f'{st_name} Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


def eval_first_sense(data):
    ''' pick first synset every time '''
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

    print(f'first sense Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


''' HELPERS '''
def get_NE_tags(sentence, seq_tagger):
    # run NER over sentence
    sentence = Sentence(sentence)
    seq_tagger.predict(sentence)
    return [label.value for label in sentence.get_labels()]

def overlap_quantity(context1, context2):
    # length of list intersection
    if len(context1) < len(context2):
        context1, context2 = context2, context1
    return len(set(context1).intersection(context2))

def get_max_NEO_synset(context, synsets):
    # synset of maximum named entity overlap with context
    return max([
        (synset[0], overlap_quantity(context, synset[1])) \
        for synset in synsets
    ], key=lambda x: x[1])


''' Dev and Test '''
def dev_pipeline(models=[
    'ner-fast', 
    'ner', 
    'ner-ontonotes-large',
    'ner-ontonotes-fast']):
    # evaluate models on training set
    for model in models:
        print(f'==============>evaluating {model}...<==============')
        eval_NEO(data=train_data, seq_tagger=Classifier.load(model), st_name=model)
        print('\n')


def test_model(model):
    print(f'==============>testing {model}...<==============')
    eval_NEO(data=test_data, seq_tagger=Classifier.load(model), st_name=model)

    
if __name__ == "__main__":
    model_name = 'ner-ontonotes-fast'
    test_model(model_name) # be patient


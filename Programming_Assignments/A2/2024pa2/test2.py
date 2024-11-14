'''IMPORT'''
from loader import main_data_loader
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from flair.data import Sentence
from flair.nn import Classifier
dev_data, test_data, dev_key, test_key = main_data_loader()
stop_words = set(stopwords.words('english'))
tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation
# load the NER tagger
NERtagger = Classifier.load('ner-fast')
import random


''' MODELS '''

''' Custom Models '''
def eval_NEO(data, keys):
    # Named Entity Overlap
    cur_sent = [None, None]
    i = 0
    for id, item in data.items():
        item_lemma = item.lemma.decode('ascii')
        s = id.split('.')[1]
        if s != cur_sent[0]:
            print(s)
            cur_sent[0] = s
            cur_sent[1] = get_NE_tags(' '.join(map(lambda x: x.decode('ascii'), item.context)))

        # def_and_examples = {}
        synsets = wn.synsets(item_lemma)
        # gather definition and examples
        synset_texts = []
        for synset in synsets:
            synset_text = ' '.join([synset.definition()] + [example for example in synset.examples()])
            # get_NE_tags(synset_text)
            # synset_texts.append((synset, synset_text))

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
        (synset, overlap_quantity(context, synset)) \
        for synset in synsets
    ], key=lambda x: x[1])


if __name__ == "__main__":
    # eval_NEO(dev_data, dev_key)
    import pandas as pd
    df = pd.DataFrame(columns=['id',  'lemma', 'context', 'index', 'label'])
    i = 0
    for id, item in dev_data.items():
        df.loc[i] = [item.id, 
                     item.lemma.decode('ascii'), 
                     ' '.join(map(lambda x: x.decode('ascii'), item.context)), 
                     item.index,
                     dev_key[item.id]
                     ]
        i += 1
    for id, item in test_data.items():
        df.loc[i] = [item.id, 
                     item.lemma.decode('ascii'), 
                     ' '.join(map(lambda x: x.decode('ascii'), item.context)), 
                     item.index,
                     test_key[item.id]
                     ]
        i += 1

    df.to_csv(sep='\t', index=False, path_or_buf=os.path.join(os.path.dirname(__file__), './pa2data.tsv'))


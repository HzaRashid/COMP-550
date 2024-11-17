'''******************************************************************'''
'''******************************************************************'''
'''
INSTRUCTIONS:
    -  Scroll to the bottom (__name__ == "__main__")
    -  run test_model(model) with the model thats already there
    
'''
'''******************************************************************'''
'''******************************************************************'''

'''IMPORT'''
from flair.embeddings import TransformerDocumentEmbeddings as TDE
from flair.embeddings import TransformerWordEmbeddings as TWE
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from flair.data import Sentence
from ast import literal_eval
import pandas as pd
import numpy as np
import os


''' load data '''
data = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'pa2data.tsv'),
    converters={'label':literal_eval},
    sep='\t'
    )

train_data = data[data['id'].str.startswith('d001')]
test_data = data[~data['id'].str.startswith('d001')]
# tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation

''' model evaluation '''
def eval_embedding_L2(data, embedder):
    ''' 
    L2 distance between transformer embeddings
    of context and synset example (if available) or definition text
    '''
    correct_prediction_ct = 0
    cur_sent = [None, None] # document-id, context-embedding
    
    for _, item in data.iterrows():
        s = item.id.split('.')[1] # document-id
        if s != cur_sent[0]: # new document
            cur_sent[0] = s
            cur_sent[1] = Sentence(item.context.replace('_', ' '))
            embedder.embed(cur_sent[1])
            # # the code below extracts the token embedding for the target lemma
            # # (the models perform worse with this)
            # sent = Sentence(item.context.replace('_', ' '))
            # word_embedder = TWE(embedder.base_model_name)
            # word_embedder.embed(sent)
            # for token in sent:
            #     if token.text != item.lemma: continue
            #     cur_sent[1] = token
            #     break
                # print(token.embedding)

        synsets = wn.synsets(item.lemma)
        tagged_synsets = []
        for ss in synsets:
            ss_text = Sentence(ss.examples()[0] if ss.examples() else ss.definition())
            embedder.embed(ss_text)
            tagged_synsets.append(
                (ss, np.linalg.norm(cur_sent[1].embedding - ss_text.embedding))
                )
            
        sense = None
        closest_ss = min(tagged_synsets, key=lambda x: x[1])[0]

        for lemma in closest_ss.lemmas():
            if lemma.name().lower() == item.lemma.lower(): 
                sense = lemma.key()

        if sense in item.label:
            correct_prediction_ct += 1

    print(f'{embedder.base_model_name} Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


''' Dev and Test '''
def dev_pipeline(models={
    'bert-base-cased': 12, # number of layers
    'sentence-transformers/all-mpnet-base-v2': 12,
    'sentence-transformers/all-MiniLM-L6-v2': 6
    }):

    for m in models:
        for layer in range(1, models[m] + 1):
            print(f'==============>evaluating {m}, layer {layer}...<==============')
            eval_embedding_L2(train_data, 
                              TDE(m, layers=str(layer), layer_mean=False)
                                )
        print(f'==============>evaluating {m}, all layers...<==============')
        eval_embedding_L2(train_data, 
                        TDE(m, layers='all', layer_mean=True))
        print('\n')

def test_model(model):
    print(f'Testing {model.base_model_name}...')
    eval_embedding_L2(test_data, model)

if __name__ == "__main__":
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = TDE(model_name)
    test_model(model) # be patient
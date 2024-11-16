'''IMPORT'''
from flair.embeddings import TransformerDocumentEmbeddings as tde
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
tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation

def eval_embedding_L2(data, embedder):
    ''' 
    L2 distance between transformer embeddings
    of context and synset example (if available) or definition text
    '''
    correct_prediction_ct = 0
    cur_sent = [None, None] # document-id, context-embedding
    
    for _, item in data.iterrows():
        s = item.id.split('.')[1]
        if s != cur_sent[0]:
            cur_sent[0] = s
            cur_sent[1] = Sentence(item.context.replace('_', ' '))
            embedder.embed(cur_sent[1])

        synsets = wn.synsets(item.lemma)
        tagged_synsets = []
        for ss in synsets:
            ss_text = Sentence(ss.examples()[0] if ss.examples() else ss.definition())
            embedder.embed(ss_text)
            tagged_synsets.append(
                (ss, np.linalg.norm(cur_sent[1].embedding - ss_text.embedding))
                )
        # print(tagged_synsets)
        sense = None
        closest_ss = min(tagged_synsets, key=lambda x: x[1])[0]

        for lemma in closest_ss.lemmas():
            if lemma.name().lower() == item.lemma.lower(): 
                sense = lemma.key()

        if sense in item.label:
            correct_prediction_ct += 1

    print(f'{embedder.base_model_name} Accuracy: {100 * float(correct_prediction_ct / len(data))}%')



def dev_pipeline(models={
    # 'bert-base-cased': 12, # number of layers
    'sentence-transformers/all-mpnet-base-v2': 12,
    'sentence-transformers/all-MiniLM-L6-v2': 6
    }):

    for m in models:
        for layer in range(1, models[m] + 1):
            print(f'==============>{m}, layer {layer}...<==============')
            eval_embedding_L2(train_data, 
                              tde(m, layers=str(layer), layer_mean=False)
                                )
        print(f'==============>{m}, all layers...<==============')
        eval_embedding_L2(train_data, 
                        tde(m, layers='all', layer_mean=True)
)

if __name__ == "__main__":
    dev_pipeline()
    # eval_embedding_L2(train_data, embedder)
    # import numpy as np
    # # init embedding
    # embedding = tde('sentence-transformers/all-MiniLM-L6-v2')
    # print(
    #     embedding.base_model_name
    # )

    # # create a sentence
    # sentence1 = Sentence('hello from here')
    # sentence2 = Sentence('goodbye from there')
    # embedding.embed(sentence1)
    # embedding.embed(sentence2)

    # print(sentence1.embedding == sentence2.embedding)

    # for s in (sentence1, sentence2):
    #     print(s.embedding)
    #     print(len(s.embedding))
    #     print(type(s.embedding))

    # print(
    #     np.linalg.norm(sentence1.embedding - sentence2.embedding)
    # )


    # embed words in sentence
    # embedding.embed(sentence)
    # now check out the embedded tokens.
    # for token in sentence:
    #     print(token)
    #     print(len(token.embedding))
    #     print(token.embedding)

    # # Example text
    # sample_text = "The quick brown fox jumps over the lazy dog"

    # # Find all parts of speech in above sentence
    # tagged = [('', x[1]) for x in pos_tag(word_tokenize(sample_text))]

    # # Extract all parts of speech from any text
    # chunker = RegexpParser("""
    #                     NP: {<DT>?<JJ>*<NN>} #To extract Noun Phrases
    #                     P: {<IN>}			 #To extract Prepositions
    #                     V: {<V.*>}			 #To extract Verbs
    #                     PP: {<p> <NP>}		 #To extract Prepositional Phrases
    #                     VP: {<V> <NP|PP>*}	 #To extract Verb Phrases
    #                     """)

    # # Print all parts of speech in above sentence
    # output = chunker.parse(tagged)

    # print("After Extracting\n", output)
    # print('', str(output))
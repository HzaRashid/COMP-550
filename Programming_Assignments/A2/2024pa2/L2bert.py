'''IMPORT'''
import os
from nltk.corpus import wordnet as wn
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
import pandas as pd
from ast import literal_eval
import numpy as np

''' load data '''
data = pd.read_csv(
    sep='\t', 
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'pa2data.tsv'),
    converters={'label':literal_eval}
    )
train_data = data[data['id'].str.startswith('d001')]

''' get pre-trained model for embeddings'''
embedding = TransformerDocumentEmbeddings('bert-base-uncased')


def eval_L2_bert(data):
    ''' 
    L2 distance between context and 
    synset example (if available) or definition 
    embeddings
    '''
    correct_prediction_ct = 0
    cur_sent = [None, None]
    
    for _, item in data.iterrows():
        s = item.id.split('.')[1]
        if s != cur_sent[0]:
            print(s)
            cur_sent[0] = s
            cur_sent[1] = Sentence(item.context)
            embedding.embed(cur_sent[1])

        synsets = wn.synsets(item.lemma)
        tagged_synsets = []
        for ss in synsets:
            ss_text = Sentence(ss.examples()[0] if ss.examples() else ss.definition())
            embedding.embed(ss_text)
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

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')




if __name__ == "__main__":
    eval_L2DS(train_data)
    # import numpy as np
    # # init embedding
    # embedding = TransformerDocumentEmbeddings('bert-base-uncased')

    # # create a sentence
    # sentence1 = Sentence('hello from here')
    # sentence2 = Sentence('hello from here')
    # embedding.embed(sentence1)
    # embedding.embed(sentence2)


    # print(sentence1.embedding)
    # print(sentence1.embedding)
    # print(len(sentence1.embedding))
    # print(type(sentence1.embedding))

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
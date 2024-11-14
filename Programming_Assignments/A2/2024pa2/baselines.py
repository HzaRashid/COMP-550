'''IMPORT'''
from loader import main_data_loader
import os
# # Download nltk data
download_dir = f"{os.getenv('VIRTUAL_ENV')}/lib/nltk_data"
# import nltk
# nltk.download('wordnet', download_dir=download_dir)
# nltk.download('stopwords', download_dir=download_dir)
# nltk.download('punkt', download_dir=download_dir)
# nltk.download('punkt_tab', download_dir=download_dir)
# #
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


dev_data, test_data, dev_key, test_key = main_data_loader()
stop_words = set(stopwords.words('english'))
tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation
# lemmatize = WordNetLemmatizer().lemmatize

''' MODELS '''

''' Custom Models '''
def eval_NEO(data, keys):
    # Named Entity Overlap
    cur_sent = [None, None]
    for id, item in data.items():
        item_lemma = item.lemma.decode('ascii')
        if id != cur_sent[0]:
            cur_sent[0] = id
            cur_sent[1] = ' '.join(map(lambda x: x.decode('ascii'), item.context))

        def_and_examples = {}
        synsets = wn.synsets(item_lemma)
        # gather definition and examples
        for synset in synsets:
            def_and_examples[synset] = []
            def_and_examples[synset].append(synset.definition())
            for example in synset.examples():
                def_and_examples[synset].append(example)

        print(def_and_examples)


''' Baseline Models '''
def eval_lesk(data, keys):
    # Lesk (using WordNet)
    correct_prediction_ct = 0
    cur_sent = [None, None]

    for _id in data:
        s = _id.split('.')[1]
        if s != cur_sent[0]: 
            cur_sent[0] = s
            cur_sent[1] = tokenize(' '.join(
                filter(lambda x: x.lower() not in stop_words,
                    map(lambda x: x.decode('ascii'), data[_id].context)
                    )).replace('_', ' '))
            
        x = data[_id].lemma.decode('ascii')
        synset = lesk(context_sentence=cur_sent[1], 
                      ambiguous_word=x)
        sense = None

        for lemma in synset.lemmas():
            if lemma.name().lower() == x.lower():
                sense = lemma.key()
                break
            
        if sense in keys[_id]:
            correct_prediction_ct += 1

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


def eval_mfs(data, keys):
    # Most Frequent Sense (using pretrained frequencies)
    correct_prediction_ct = 0
    for doc_id in data:
        freq2sense = {}
        x = data[doc_id].lemma.decode('ascii')
        for synset in wn.synsets(x):
            x_sense = None
            synsetfq = 0
            for lemma in synset.lemmas():
                synsetfq += lemma.count()
                if lemma.name().lower() == x.lower(): 
                    x_sense = lemma.key()

            if synsetfq not in freq2sense: 
                freq2sense[synsetfq] = []
            freq2sense[synsetfq].append(x_sense)
            
        if set(freq2sense[max(freq2sense)]).intersection(keys[doc_id]):
            correct_prediction_ct += 1

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


''' HELPERS '''
def get_NE_tags(sentence):
    return set(label.value for label in sentence.get_labels())

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
    eval_NEO(dev_data, dev_key)
    # eval_mfs(test_data, test_key)
    # eval_lesk(test_data, test_key)

    # x = 'North_America'
    # for synset in wn.synsets(x):
    #     print(synset, synset.definition())

    # for synset in wn.synsets('car'):
    #     print(synset, synset.definition())

    #     for lemma in synset.lemmas():
    #         print(lemma, lemma.hypernyms())

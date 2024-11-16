'''IMPORT'''
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from loader import main_data_loader
from nltk.wsd import lesk 

dev_data, test_data, dev_key, test_key = main_data_loader()
stop_words = set(stopwords.words('english'))
tokenize = RegexpTokenizer(r'\w+').tokenize # removes punctuation

''' Evaluate baseline models '''
def eval_lesk(data, keys):
    # Lesk (using WordNet)
    correct_prediction_ct = 0
    cur_sent = [None, None] # document-id, tokenized-context

    for id, item in data.items():
        s = id.split('.')[1] # document-id
        if s != cur_sent[0]: 
            cur_sent[0] = s
            cur_sent[1] = preprocess(item.context)
            
        target = item.lemma.decode('ascii') # disambiguation target
        synset = lesk(context_sentence=cur_sent[1], 
                      ambiguous_word=target) # get synset of maximum word overlap
        
        target_sense = None # get target sense
        for lemma in synset.lemmas():
            if lemma.name().lower() == target.lower():
                target_sense = lemma.key()
                break
            
        if target_sense in keys[id]: # compare the Lesk-sense against gold label(s)
            correct_prediction_ct += 1

    print(f'Lesk Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


def eval_mfs(data, keys):
    # Most Frequent Sense (using pre-trained frequencies)
    correct_prediction_ct = 0
    for id, item in data.items():
        freq2sense = {} # maps integer-frequency to synset
        target = item.lemma.decode('ascii')
        for ss in wn.synsets(target):
            target_sense = None # sense of target word w.r.t synset
            sfq = 0 # sum the synset's lemma frequencies
            for lemma in ss.lemmas():
                sfq += lemma.count()
                if lemma.name().lower() == target.lower(): 
                    target_sense = lemma.key()

            if sfq not in freq2sense: freq2sense[sfq] = []
            freq2sense[sfq].append(target_sense)

        # M.F.S step: intersect most frequent sense(s) with gold label(s)
        if set(freq2sense[max(freq2sense)]).intersection(keys[id]):
            correct_prediction_ct += 1

    print(f'M.F.S Accuracy: {100 * float(correct_prediction_ct / len(data))}%')


''' Helpers '''
def preprocess(text):
    return tokenize(' '.join(
        filter(lambda x: x.lower() not in stop_words,
               map(lambda x: x.decode('ascii'), text)
               )).replace('_', ' '))

if __name__ == "__main__":
    eval_lesk(test_data, test_key)
    eval_mfs(test_data, test_key)
    


    '''
    the following code was used 
    to download the nltk data:

    # Download nltk data
    download_dir = f"{os.getenv('VIRTUAL_ENV')}/lib/nltk_data"
    import nltk
    nltk.download('wordnet', download_dir=download_dir)
    nltk.download('stopwords', download_dir=download_dir)
    nltk.download('punkt', download_dir=download_dir)
    nltk.download('punkt_tab', download_dir=download_dir)
    #
    '''

    # x = 'North_America'
    # for synset in wn.synsets(x):
        # print(synset, synset.definition())
        # print([lemma.name() for lemma in synset.lemmas()])

    # for synset in wn.synsets('car'):
    #     print(synset, synset.definition())

    #     for lemma in synset.lemmas():
    #         print(lemma, lemma.hypernyms())

'''IMPORT'''
from loader import main_data_loader
import os
# # Download nltk data
# download_dir = f"{os.getenv('VIRTUAL_ENV')}/lib/nltk_data"
# import nltk
# nltk.download('wordnet', download_dir=download_dir)
# nltk.download('stopwords', download_dir=download_dir)
# nltk.download('punkt', download_dir=download_dir)
# nltk.download('punkt_tab', download_dir=download_dir)
# #
from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer


dev_data, test_data, dev_key, test_key = main_data_loader()
# print(len(dev_data), len(test_data))
# stop_words = set(stopwords.words('english'))


def eval_mfs(data, keys):
    correct_prediction_ct = 0
    for document_id in data:
        freqs = {}
        x = data[document_id].lemma.decode('ascii')
        for synset in wn.synsets(x):
            x_sense = None
            synset_freq = 0
            for lemma in synset.lemmas():
                synset_freq += lemma.count()
                if lemma.name().lower() == x.lower(): 
                    x_sense = lemma.key()

            if synset_freq not in freqs: 
                freqs[synset_freq] = []
            freqs[synset_freq].append(x_sense)
            
        if set(freqs[max(freqs)]).intersection(keys[document_id]):
            correct_prediction_ct += 1

    print(f'Accuracy: {100 * float(correct_prediction_ct / len(data))}%')

if __name__ == "__main__":
    eval_mfs(test_data, test_key)
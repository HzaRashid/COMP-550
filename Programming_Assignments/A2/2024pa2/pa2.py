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
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""
dev_data, test_data, dev_key, test_key = main_data_loader()
# print(len(train_data), len(test_data))
stop_words = set(stopwords.words('english'))
sentence_tokens = word_tokenize(example_sent)

print(sentence_tokens)
correct_prediction_ct = 0
i = 0
for document in dev_data:
    x = dev_data[document].lemma.decode('ascii')

    synsets = wordnet.synsets(x)

    sense_freqs = {}
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.name().lower() != x.lower(): continue
            if lemma.count() not in sense_freqs:
                sense_freqs[lemma.count()] = []
            sense_freqs[lemma.count()].append(lemma.key())
            print(lemma._key)

    max_freq = max(sense_freqs)
    for sense_key in sense_freqs[max(sense_freqs)]:
        if sense_key == dev_data[document]:
            # correct_prediction_ct += 1
            break


print(correct_prediction_ct)
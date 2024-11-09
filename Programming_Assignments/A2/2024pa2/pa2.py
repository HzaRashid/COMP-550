'''IMPORT'''

from loader import main_data_loader
import os
# # Download nltk submodules
# download_dir = f"{os.getenv('VIRTUAL_ENV')}/lib/nltk_data"
# import nltk
# nltk.download('wordnet', download_dir=download_dir)
# #
from nltk.corpus import wordnet


# train_data, test_data = main_data_loader()
# print(len(train_data), len(test_data))

test_str = "Latin_america".lower()

synsets = wordnet.synsets(test_str)

sense_freqs = {}
for synset in synsets:
    # print("SYNSET:", synset.name())
    if synset.name()[-2:] != '01': continue

    for lemma in synset.lemmas():
        if lemma.name().lower() != test_str: continue
        sense_freqs[lemma.key()] = lemma.count()

max_freq = max(sense_freqs.values())
print(sense_freqs)

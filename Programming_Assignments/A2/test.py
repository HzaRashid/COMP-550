from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('a motor vehicle with four wheels; usually propelled by an internal combustion engine.')

# load the NER tagger
tagger = Classifier.load('ner-ontonotes-large')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)

print(sentence.get_labels())

def get_NE_tags(sentence):
    return set(label.value for label in sentence.get_labels())

print(get_NE_tags(sentence))
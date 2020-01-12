---
title: "Part of Speech Tagging"
description: "Part of Speech tagging using Hidden Markov Model"
categories: [machinelearning]
tags: [python, machinelearning]
---

# Part of Speech Tagging

## Problem Statement

In this article, we'll try and predict the Part of Speech (POS) tag for each word in a provided sentence.

Here we will build a model using Hidden Markov Models which would help us predict the POS tags for all words in an utterance.

### What is a POS tag?

In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging or word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context—i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. A simplified form of this is commonly taught to school-age children, in the identification of words as nouns, verbs, adjectives, adverbs, etc.

### Dataset

The dataset can be downloaded from <a href="https://drive.google.com/open?id=1wNC_Oe99AqwaaCK0slF4zS64pq-4OY27"> here </a>. 

#### Dataset Description
##### Sample Tuple
b100-5507

Mr.	NOUN
<br>
Podger	NOUN
<br>
had	VERB
<br>
thanked	VERB
<br>
him	PRON
<br>
gravely	ADV
<br>
,	.
<br>
and	CONJ
<br>
now	ADV
<br>
he	PRON
<br>
made	VERB
<br>
use	NOUN
<br>
of	ADP
<br>
the	DET
<br>
advice	NOUN
<br>
.	.
<br>
##### Explanation
The first token "b100-5507" is just a key and acts like an identifier to indicate the beginning of a sentence.
<br>
The other tokens have a (Word, POS Tag) pairing.

__List of POS Tags are:__
.
<br>
ADJ
<br>
ADP
<br>
ADV
<br>
CONJ
<br>
DET
<br>
NOUN
<br>
NUM
<br>
PRON
<br>
PRT
<br>
VERB
<br>
X

__Note__
<br>
__.__ is used to indicate special characters such as '.', ','
<br>
__X__ is used to indicate vocab not part of Enlish Language mostly.
Others are Standard POS tags.

### Train-Test Split
Let us use a 80-20 split of our data for training and evaluation purpose.




```python
#Import libraries
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict, namedtuple, OrderedDict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
import os
from io import BytesIO
from itertools import chain
import random
```


```python
#Read data
Sentence = namedtuple("Sentence", "words tags")

def read_data(filename):
#   Read tagged data from the file provided.
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags( ):
#   Read all the tags mentioned above
    tags = ('.',
           'ADJ',
           'ADP',
           'ADV',
           'CONJ',
           'DET',
           'NOUN',
           'NUM',
           'PRON',
           'PRT',
           'VERB',
           'X')
    return frozenset(tags)

# Class to divide the dataset into train and test, based on the keys.
class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


#     Main class to read the dataset and split it into train and test set
class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags()
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())
    

```


```python
#Pre-process data (Whatever you feel might be required)
data = Dataset("data.txt", train_test_split=0.8)

print("There are {} sentences in the corpus.".format(len(data)))
print("There are {} sentences in the training set.".format(len(data.training_set)))
print("There are {} sentences in the testing set.".format(len(data.testing_set)))

assert len(data) == len(data.training_set) + len(data.testing_set), \
       "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"
```

    There are 57340 sentences in the corpus.
    There are 45872 sentences in the training set.
    There are 11468 sentences in the testing set.
    


```python
#Data Description

# Taking a key at random from dataset to check the sentence
key = 'b100-38532'
print("Sentence: {}".format(key))
print("words:\n\t{!s}".format(data.sentences[key].words))
print("tags:\n\t{!s}".format(data.sentences[key].tags))


# Check number of unique words in the corpus
print("There are a total of {} samples of {} unique words in the corpus."
      .format(data.N, len(data.vocab)))

# Number of samples generated using unique words in training set.
print("There are {} samples of {} unique words in the training set."
      .format(data.training_set.N, len(data.training_set.vocab)))

# Number of the samples generated using unique words in testing set.
print("There are {} samples of {} unique words in the testing set."
      .format(data.testing_set.N, len(data.testing_set.vocab)))

# Number of words which are not present in the training set.
print("There are {} words in the test set that are missing in the training set."
      .format(len(data.testing_set.vocab - data.training_set.vocab)))

assert data.N == data.training_set.N + data.testing_set.N, \
       "The number of training + test samples should sum to the total number of samples"

# accessing words with Dataset.X and tags with Dataset.Y 
for i in range(2):    
    print("Sentence {}:".format(i + 1), data.X[i])
    print()
    print("Labels {}:".format(i + 1), data.Y[i])
    print()
```

    Sentence: b100-38532
    words:
    	('Perhaps', 'it', 'was', 'right', ';', ';')
    tags:
    	('ADV', 'PRON', 'VERB', 'ADJ', '.', '.')
    There are a total of 1161192 samples of 56057 unique words in the corpus.
    There are 928458 samples of 50536 unique words in the training set.
    There are 232734 samples of 25112 unique words in the testing set.
    There are 5521 words in the test set that are missing in the training set.
    Sentence 1: ('Mr.', 'Podger', 'had', 'thanked', 'him', 'gravely', ',', 'and', 'now', 'he', 'made', 'use', 'of', 'the', 'advice', '.')
    
    Labels 1: ('NOUN', 'NOUN', 'VERB', 'VERB', 'PRON', 'ADV', '.', 'CONJ', 'ADV', 'PRON', 'VERB', 'NOUN', 'ADP', 'DET', 'NOUN', '.')
    
    Sentence 2: ('But', 'there', 'seemed', 'to', 'be', 'some', 'difference', 'of', 'opinion', 'as', 'to', 'how', 'far', 'the', 'board', 'should', 'go', ',', 'and', 'whose', 'advice', 'it', 'should', 'follow', '.')
    
    Labels 2: ('CONJ', 'PRT', 'VERB', 'PRT', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'ADP', 'ADV', 'ADV', 'DET', 'NOUN', 'VERB', 'VERB', '.', 'CONJ', 'DET', 'NOUN', 'PRON', 'VERB', 'VERB', '.')
    
    


```python
#HMM Model Goes Here

# To determine the probability of one state finite automata(single tag probability)
def unigram_counts(sequences):

    return Counter(sequences)

tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
tag_unigrams = unigram_counts(tags)


# To determine the probability of two states combined to see which tags comes together
# with high probability
def bigram_counts(sequences):

    d = Counter(sequences)
    return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]
o = [(tags[i],tags[i+1]) for i in range(0,len(tags)-2,2)]
tag_bigrams = bigram_counts(o)


# For transition probabilities, starting and ending tags are added.

def starting_counts(sequences):
    
    d = Counter(sequences)
    return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]
starts_tag = [i[0] for i in data.Y]
tag_starts = starting_counts(starts_tag)


def ending_counts(sequences):
    
    d = Counter(sequences)
    return d

end_tag = [i[len(i)-1] for i in data.Y]
tag_ends = ending_counts(end_tag)


def pair_counts(tags, words):
    d = defaultdict(lambda: defaultdict(int))
    for tag, word in zip(tags, words):
        d[tag][word] += 1
        
    return d
# 

basic_model = HiddenMarkovModel(name="base-hmm-tagger")

tags = [tag for i, (word, tag) in enumerate(data.stream())]
words = [word for i, (word, tag) in enumerate(data.stream())]

tags_count=unigram_counts(tags)
tag_words_count=pair_counts(tags,words)

starting_tag_list=[i[0] for i in data.Y]
ending_tag_list=[i[-1] for i in data.Y]

starting_tag_count=starting_counts(starting_tag_list)#the number of times a tag occured at the start
ending_tag_count=ending_counts(ending_tag_list)      #the number of times a tag occured at the end



to_pass_states = []
for tag, words_dict in tag_words_count.items():
    total = float(sum(words_dict.values()))
    distribution = {word: count/total for word, count in words_dict.items()}
    tag_emissions = DiscreteDistribution(distribution)
    tag_state = State(tag_emissions, name=tag)
    to_pass_states.append(tag_state)


basic_model.add_states()    
    
# Calculate starting tag probabity and add it to transition prob matrix of the model.
start_prob={}

for tag in tags:
    start_prob[tag]=starting_tag_count[tag]/tags_count[tag]

for tag_state in to_pass_states :
    basic_model.add_transition(basic_model.start,tag_state,start_prob[tag_state.name])    


# Calculate ending tag probabity and add it to transition prob matrix of the model.
end_prob={}

for tag in tags:
    end_prob[tag]=ending_tag_count[tag]/tags_count[tag]
for tag_state in to_pass_states :
    basic_model.add_transition(tag_state,basic_model.end,end_prob[tag_state.name])
    

# Calculate bigram probability and add it to transition prob matrix of the model.
transition_prob_pair={}

for key in tag_bigrams.keys():
    transition_prob_pair[key]=tag_bigrams.get(key)/tags_count[key[0]]
for tag_state in to_pass_states :
    for next_tag_state in to_pass_states :
        basic_model.add_transition(tag_state,next_tag_state,transition_prob_pair[(tag_state.name,next_tag_state.name)])

#  Generate the model.
basic_model.bake()
```


```python
#Model Accuracy Evaluation

# For any unknown word in dictionary of Training set, replace it with 'nan'
def replace_unknown(sequence):
    
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]


# Using viterbi algorithm to decode the model.
def simplify_decoding(X, model):
    
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]

def accuracy(X, Y, model):
    
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions


hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))
```

    training accuracy basic hmm model: 97.49%
    testing accuracy basic hmm model: 96.09%
    


```python
#Adds code blocks wherever you feel necessary
for key in data.testing_set.keys[:2]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")
```

    Sentence Key: b100-28144
    
    Predicted labels:
    -----------------
    ['CONJ', 'NOUN', 'NUM', '.', 'NOUN', 'NUM', '.', 'NOUN', 'NUM', '.', 'CONJ', 'NOUN', 'NUM', '.', '.', 'NOUN', '.', '.']
    
    Actual labels:
    --------------
    ('CONJ', 'NOUN', 'NUM', '.', 'NOUN', 'NUM', '.', 'NOUN', 'NUM', '.', 'CONJ', 'NOUN', 'NUM', '.', '.', 'NOUN', '.', '.')
    
    
    Sentence Key: b100-23146
    
    Predicted labels:
    -----------------
    ['PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'VERB', '.', 'ADP', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', '.']
    
    Actual labels:
    --------------
    ('PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'VERB', '.', 'ADP', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', '.')
    
    
    

### Happy Coding!
